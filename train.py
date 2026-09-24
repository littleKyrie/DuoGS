#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
import torch
from random import randint
from utils.loss_utils import l1_loss
from utils.loss_utils import fast_ssim as ssim
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, LossParamsS2, LossParamsS1
import shutil
from utils.warp_utils import Warpper
from utils.graph_utils import node_graph
from rich.console import Console

CONSOLE = Console(width=120)
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/",args.source_path.split('/')[-1] + '_' + unique_str[0:2])

        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    return tb_writer

def training_joint(dataset, opt, pipe, lossp, testing_iterations, debug_from, is_start_frame, frame_idx = 1, args = None):
    CONSOLE.log("Training joint:", frame_idx)
    parallel_load = args.parallel_load
    subseq_iters = args.subseq_iters
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    ply_path = args.ply_path

    # Initialize the model(Gaussians in this project)
    # Clarify the main training params of the model.
    joint_gaussian = GaussianModel(dataset.sh_degree)

    # Load training data(cameras\images\ply in this project)
    scene = Scene(dataset, joint_gaussian, dynamic_training = True, load_frame_id = frame_idx, ply_path = ply_path, parallel_load=parallel_load, stage = 1, warpDQB=warpDQB)

    # Set up the training parameters for the model and add them to the optimizer
    # Build the connection between the model params and the optimizer.
    joint_gaussian.training_setup_t1(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    if is_start_frame==False:
        opt.iterations = subseq_iters if subseq_iters else (opt.iterations // 2)

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    loss = 0
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        joint_gaussian.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            joint_gaussian.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_canonical = render(viewpoint_cam, joint_gaussian, pipe, background)
        image_canonical , viewspace_point_tensor, visibility_filter, radii = render_canonical["render"], render_canonical["viewspace_points"], render_canonical["visibility_filter"], render_canonical["radii"]
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image_canonical, gt_image)
        if is_start_frame:
            ssim_value = ssim(image_canonical, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        else:
            loss = Ll1

        loss_info = {}
        loss_info["rgb_loss"] = loss.item()
        reg_loss, reg_loss_info = joint_graph.compute_loss(joint_gaussian, lossp, is_start_frame)
        loss_info.update(reg_loss_info)

        loss = loss + reg_loss
        loss.backward()

        iter_end.record()
        if is_start_frame:
            loss_info['gs_num'] = joint_gaussian.get_xyz.shape[0]
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 100 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.7f}", **{k: f"{v:.7f}" for k, v in loss_info.items()}})
                progress_bar.update(100)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, joint_gaussian, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))

            if is_start_frame and iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                joint_gaussian.max_radii2D[visibility_filter] = torch.max(joint_gaussian.max_radii2D[visibility_filter], radii[visibility_filter])
                joint_gaussian.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    joint_gaussian.densify_and_prune(opt.densify_grad_threshold, 0.1, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    joint_gaussian.reset_opacity()
                
            if iteration == opt.densify_until_iter and is_start_frame:
                points_num = joint_gaussian._xyz.shape[0]
                prune_scale = int(points_num // opt.joint_gs_num)
                prune_scale = max(prune_scale, 1)
                prune_mask = torch.ones(points_num, dtype=torch.bool, device=joint_gaussian._xyz.device)
                prune_mask[::prune_scale] = False
                joint_gaussian.prune_points(prune_mask)

            if iteration < opt.iterations:
                if  is_start_frame==False:
                    joint_gaussian.lock_gradient( lock_opacity = True, lock_scaling= True, lock_features= True)       
                joint_gaussian.optimizer.step()
                joint_gaussian.optimizer.zero_grad(set_to_none = True)

    joint_gaussian.save_ply(os.path.join(dataset.model_path, "ckt", "point_cloud_%d.ply"% (frame_idx )))
    
    if is_start_frame:
        joint_graph.graph_init(joint_gaussian.get_xyz, k = 8)
    
    joint_graph.regular_term_setup(joint_gaussian, velocity_option=True, warpDQB=warpDQB)
    



def training_skin(dataset, opt, pipe, lossp, testing_iterations, debug_from, is_start_frame, frame_idx = 1, args = None):
    CONSOLE.log("Training skin:", frame_idx)
    ply_path = args.ply_path
    parallel_load = args.parallel_load
    subseq_iters = args.subseq_iters
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    
    skin_gaussian = GaussianModel(dataset.sh_degree)

    if not is_start_frame:
        warpDQB.loadMotion(args.motion_folder, frame_idx,  sequential = args.seq, start_frame=args.frame_st)

    scene = Scene(dataset, skin_gaussian, dynamic_training = True, load_frame_id = frame_idx, ply_path = ply_path, parallel_load=parallel_load, stage = 2, warpDQB = warpDQB, seq=args.seq)

    if is_start_frame == False and subseq_iters:
            opt.position_lr_max_steps = subseq_iters

    if is_start_frame:
        skin_gaussian = GaussianModel(dataset.sh_degree)
        skin_gaussian.load_ply(os.path.join(args.motion_folder, "ckt", "point_cloud_%d.ply") % frame_idx, scene.cameras_extent)
        CONSOLE.log("Number of points at initialization: ", skin_gaussian._xyz.shape[0])
        

    if is_start_frame :
        skin_gaussian.training_setup_t2(opt)
    else:
        skin_gaussian.training_setup_t2_control(opt, warpDQB)


    print("number of gaussians:", len(skin_gaussian.get_xyz))
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    if is_start_frame == False:
        opt.iterations = subseq_iters if subseq_iters else (opt.iterations // 2)

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        skin_gaussian.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            skin_gaussian.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_canonical = render(viewpoint_cam, skin_gaussian, pipe, background)
        image_canonical , viewspace_point_tensor, visibility_filter, radii = render_canonical["render"], render_canonical["viewspace_points"], render_canonical["visibility_filter"], render_canonical["radii"]
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image_canonical, gt_image)
        ssim_value = ssim(image_canonical, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        loss_info = {}
        loss_info["rgb_loss"] = loss.item()
        reg_loss, reg_loss_info = skin_graph.compute_loss(skin_gaussian, lossp, is_start_frame)
        loss_info.update(reg_loss_info)
        loss = loss + reg_loss        
        loss.backward()
        iter_end.record()

        if is_start_frame:
            loss_info['gs_num'] = skin_gaussian.get_xyz.shape[0]

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 100 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.7f}", **{k: f"{v:.7f}" for k, v in loss_info.items()}})
                progress_bar.update(100)
            if iteration == opt.iterations:
                progress_bar.close()
            
            training_report(tb_writer, skin_gaussian, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))

            # first frame Densification
            if is_start_frame and iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                skin_gaussian.max_radii2D[visibility_filter] = torch.max(skin_gaussian.max_radii2D[visibility_filter], radii[visibility_filter])
                skin_gaussian.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    skin_gaussian.densify_and_prune(opt.densify_grad_threshold, 0.005 * 3, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    skin_gaussian.reset_opacity()
            
            if iteration < opt.iterations:
                    skin_gaussian.optimizer.step()
                    skin_gaussian.optimizer.zero_grad(set_to_none = True)
                    
        if not is_start_frame:
            if args.seq:
                warpDQB.warping(skin_gaussian, skin_gaussian.raw_xyz, skin_gaussian.raw_rot)
            else:
                warpDQB.warping(skin_gaussian, warpDQB.raw_xyz, warpDQB.raw_rot)
                
    skin_gaussian.save_ply(os.path.join(dataset.model_path, "ckt", "point_cloud_%d.ply"% (frame_idx )))
    
    if is_start_frame:
        warpDQB.record_gaussian(skin_gaussian)
        skin_graph.graph_init(skin_gaussian.get_xyz)
        
    skin_graph.regular_term_setup(skin_gaussian, velocity_option=False)
    if not is_start_frame:
        warpDQB.save_motion(frame_idx, os.path.join(dataset.model_path, "joint_opt"))



def training_report(tb_writer, gaussian, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, gaussian, *renderArgs)["render"], 0.0, 1.0)

                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                CONSOLE.log("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up argument parser
    parser = ArgumentParser(description="Training script parameters")
    # Add preset parameters to the parser
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    lossp1 = LossParamsS1(parser)
    lossp2 = LossParamsS2()

    # Add custom parameters to the parser which can be set from the command line
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[15000, 30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--frame_st", type=int, default=110)
    parser.add_argument("--frame_ed", type=int, default=200)
    parser.add_argument("--frame_step", type=int, default=1)
    parser.add_argument('--motion_folder', type=str, default= None)
    parser.add_argument('--ply_path', type=str, default= None)
    parser.add_argument('--seq', action='store_true', default=False)
    parser.add_argument('--parallel_load', action='store_true', default=False)
    parser.add_argument("--subseq_iters", type=int, default=None)
    parser.add_argument('--training_mode', type=int, default= 0, help="0: training motion and skin, 1: motion only, 2: skin only")
    # Parse all the arguments and store them in the args
    # argv is a list containing the command line input passed to the script. 
    # argv[0] is the script name, so we skip it and parse the rest of the arguments
    args = parser.parse_args(sys.argv[1:])

    # Create output folder for saving the trained model
    model_path = str(args.model_path)
    os.makedirs(args.model_path, exist_ok = True)

    print("Optimizing " + args.model_path)

    # Copy original 3DGS files to the output folder for reproducibility
    shutil.copy('arguments/__init__.py', args.model_path)
    shutil.copy('utils/graph_utils.py', args.model_path)
    shutil.copy('train.py', args.model_path)
    shutil.copy('scene/gaussian_model.py', args.model_path)
    
    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Create output folder path for saving the trained tracking
    if args.motion_folder is None:
        args.motion_folder = os.path.join(args.model_path, 'track')

    # Control error detection for autograd when backpropgating    
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # Configure temporal warping and run frame-by-frame training
    # warpDQB.stFrame_ is always equal to the input starting frame
    # warpDQB.step_ is the step size between consecutive frames
    warpDQB = Warpper(args.frame_st, args.frame_ed, args.frame_step)

    joint_graph = node_graph()
    skin_graph = node_graph()

    frame_range = range(args.frame_st, args.frame_ed, args.frame_step)
    is_start_frame = True

    for frame_idx in frame_range:
        if args.training_mode == 0 or args.training_mode == 1:
            if args.motion_folder is None:
                args.model_path = os.path.join(model_path, 'track')
            else:
                args.model_path = args.motion_folder
            # extract the corresponding params from the parsed arguments(args) object with stored values
            training_joint(lp.extract(args), op.extract(args), pp.extract(args), lossp1.extract(args), args.test_iterations, args.debug_from, is_start_frame, frame_idx, args=args)

        if args.training_mode == 0 or args.training_mode == 2:    
            args.model_path = model_path 
            training_skin(lp.extract(args), op.extract(args), pp.extract(args), lossp2, args.test_iterations, args.debug_from, is_start_frame, frame_idx, args=args)
                        
        is_start_frame = False
        
    # All done
    print("\nTraining complete.")