# import pickle
# import random
# import logging
#
# import cv2
# import torch
# import numpy as np
# from tqdm import tqdm, trange
#
# # new imports for single-image CLI
# import argparse
# import os
# from lib.config import Config
# from lib.experiment import Experiment
#
# # python
# def infer_single(self, img_path, epoch, view=False, save_path=None):
#     """
#     Single-image inference.
#     - img_path: str path to an RGB/BGR image on disk.
#     - epoch: which trained checkpoint to load.
#     - view: if True, show a window with overlay.
#     - save_path: if set to a file path, save the overlaid image.
#     Returns decoded prediction (e.g., lanes list).
#     """
#     # 1) Load model checkpoint
#     model = self.cfg.get_model()
#     model.load_state_dict(self.exp.get_epoch_model(epoch))
#     model = model.to(self.device)
#     model.eval()
#
#     # 2) Read image
#     bgr = cv2.imread(img_path)
#     if bgr is None:
#         raise FileNotFoundError(f"Cannot read image at: {img_path}")
#     orig_h, orig_w = bgr.shape[:2]
#
#     # 3) Preprocess (reuse dataset hints if available, fallback to ImageNet)
#     ds = self.cfg.get_dataset('test')
#     H = getattr(ds, 'img_h', None) or getattr(ds, 'input_h', None)
#     W = getattr(ds, 'img_w', None) or getattr(ds, 'input_w', None)
#     mean = np.array(getattr(ds, 'mean', [0.485, 0.456, 0.406]), dtype=np.float32)
#     std  = np.array(getattr(ds, 'std',  [0.229, 0.224, 0.225]), dtype=np.float32)
#
#     rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
#     # If model expects a specific size, resize and remember scaling factors
#     if H is not None and W is not None:
#         resized = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)
#         scale_x = orig_w / float(W)
#         scale_y = orig_h / float(H)
#     else:
#         # No resizing -> identity scale
#         resized = rgb
#         H, W = resized.shape[:2]
#         scale_x = scale_y = 1.0
#
#     x = resized.astype(np.float32) / 255.0
#     x = (x - mean) / std
#     x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(self.device)
#
#     # 4) Forward + decode
#     test_params = self.cfg.get_test_parameters()
#     with torch.no_grad():
#         out = model(x, **test_params)
#         preds = model.decode(out, as_lanes=True)
#
#     # 5) Optional visualization / saving
#     if view or save_path:
#         vis = bgr.copy()
#         try:
#             # Draw decoded lanes as points, rescale to original image coordinates
#             for lane in preds[0]:
#                 for pt in lane:
#                     # pt may be (x, y) in resized image coordinates
#                     x_i = int(round(pt[0] * scale_x))
#                     y_i = int(round(pt[1] * scale_y))
#                     if 0 <= x_i < orig_w and 0 <= y_i < orig_h:
#                         cv2.circle(vis, (x_i, y_i), 2, (0, 255, 0), -1)
#         except Exception:
#             pass
#         if view:
#             cv2.imshow('pred', vis)
#             cv2.waitKey(0)
#         if save_path:
#             # ensure directory exists
#             save_dir = os.path.dirname(save_path)
#             if save_dir:
#                 os.makedirs(save_dir, exist_ok=True)
#             cv2.imwrite(save_path, vis)
#
#     # Return lanes in original image coordinates as well for convenience
#     lanes_rescaled = []
#     try:
#         for lane in preds[0]:
#             lane_rescaled = [(pt[0] * scale_x, pt[1] * scale_y) for pt in lane]
#             lanes_rescaled.append(lane_rescaled)
#     except Exception:
#         lanes_rescaled = preds[0]
#
#     return lanes_rescaled
#
# # Example usage somewhere in your app:
# runner = Runner(cfg, exp, device, view=None)
# lanes = infer_single(runner, r'C:\Users\16031\PycharmProjects\DeepRacer\lib\img.png', epoch=20, view=True, save_path=r'C:\path\to\out.jpg')
#
# class Runner:
#     def __init__(self, cfg, exp, device, resume=False, view=None, deterministic=False):
#         self.cfg = cfg
#         self.exp = exp
#         self.device = device
#         self.resume = resume
#         self.view = view
#         self.logger = logging.getLogger(__name__)
#
#         # Fix seeds
#         torch.manual_seed(cfg['seed'])
#         np.random.seed(cfg['seed'])
#         random.seed(cfg['seed'])
#
#         if deterministic:
#             torch.backends.cudnn.deterministic = True
#             torch.backends.cudnn.benchmark = False
#
#     def train(self):
#         self.exp.train_start_callback(self.cfg)
#         starting_epoch = 1
#         model = self.cfg.get_model()
#         model = model.to(self.device)
#         optimizer = self.cfg.get_optimizer(model.parameters())
#         scheduler = self.cfg.get_lr_scheduler(optimizer)
#         if self.resume:
#             last_epoch, model, optimizer, scheduler = self.exp.load_last_train_state(model, optimizer, scheduler)
#             starting_epoch = last_epoch + 1
#         max_epochs = self.cfg['epochs']
#         train_loader = self.get_train_dataloader()
#         loss_parameters = self.cfg.get_loss_parameters()
#         for epoch in trange(starting_epoch, max_epochs + 1, initial=starting_epoch - 1, total=max_epochs):
#             self.exp.epoch_start_callback(epoch, max_epochs)
#             model.train()
#             pbar = tqdm(train_loader)
#             for i, (images, labels, _) in enumerate(pbar):
#                 images = images.to(self.device)
#                 labels = labels.to(self.device)
#
#                 # Forward pass
#                 outputs = model(images, **self.cfg.get_train_parameters())
#                 loss, loss_dict_i = model.loss(outputs, labels, **loss_parameters)
#
#                 # Backward and optimize
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#
#                 # Scheduler step (iteration based)
#                 scheduler.step()
#
#                 # Log
#                 postfix_dict = {key: float(value) for key, value in loss_dict_i.items()}
#                 postfix_dict['lr'] = optimizer.param_groups[0]["lr"]
#                 self.exp.iter_end_callback(epoch, max_epochs, i, len(train_loader), loss.item(), postfix_dict)
#                 postfix_dict['loss'] = loss.item()
#                 pbar.set_postfix(ordered_dict=postfix_dict)
#             self.exp.epoch_end_callback(epoch, max_epochs, model, optimizer, scheduler)
#
#             # Validate
#             if (epoch + 1) % self.cfg['val_every'] == 0:
#                 self.eval(epoch, on_val=True)
#         self.exp.train_end_callback()
#
#     def eval(self, epoch, on_val=False, save_predictions=False):
#         model = self.cfg.get_model()
#         model_path = self.exp.get_checkpoint_path(epoch)
#         self.logger.info('Loading model %s', model_path)
#         model.load_state_dict(self.exp.get_epoch_model(epoch))
#         model = model.to(self.device)
#         model.eval()
#         if on_val:
#             dataloader = self.get_val_dataloader()
#         else:
#             dataloader = self.get_test_dataloader()
#         test_parameters = self.cfg.get_test_parameters()
#         predictions = []
#         self.exp.eval_start_callback(self.cfg)
#         with torch.no_grad():
#             for idx, (images, _, _) in enumerate(tqdm(dataloader)):
#                 images = images.to(self.device)
#                 output = model(images, **test_parameters)
#                 prediction = model.decode(output, as_lanes=True)
#                 predictions.extend(prediction)
#                 if self.view:
#                     img = (images[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
#                     img, fp, fn = dataloader.dataset.draw_annotation(idx, img=img, pred=prediction[0])
#                     if self.view == 'mistakes' and fp == 0 and fn == 0:
#                         continue
#                     cv2.imshow('pred', img)
#                     cv2.waitKey(0)
#
#         if save_predictions:
#             with open('predictions.pkl', 'wb') as handle:
#                 pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
#         self.exp.eval_end_callback(dataloader.dataset.dataset, predictions, epoch)
#
#     def get_train_dataloader(self):
#         train_dataset = self.cfg.get_dataset('train')
#         train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                                    batch_size=self.cfg['batch_size'],
#                                                    shuffle=True,
#                                                    num_workers=8,
#                                                    worker_init_fn=self._worker_init_fn_)
#         return train_loader
#
#     def get_test_dataloader(self):
#         test_dataset = self.cfg.get_dataset('test')
#         test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                                   batch_size=self.cfg['batch_size'] if not self.view else 1,
#                                                   shuffle=False,
#                                                   num_workers=8,
#                                                   worker_init_fn=self._worker_init_fn_)
#         return test_loader
#
#     def get_val_dataloader(self):
#         val_dataset = self.cfg.get_dataset('val')
#         val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
#                                                  batch_size=self.cfg['batch_size'],
#                                                  shuffle=False,
#                                                  num_workers=8,
#                                                  worker_init_fn=self._worker_init_fn_)
#         return val_loader
#
#     @staticmethod
#     def _worker_init_fn_(_):
#         torch_seed = torch.initial_seed()
#         np_seed = torch_seed // 2**32 - 1
#         random.seed(torch_seed)
#         np.random.seed(np_seed)
#
#
# # CLI entry point to run single-image inference using experiment checkpoints
# def main():
#     parser = argparse.ArgumentParser(description='Single-image lane inference using saved Experiment checkpoints')
#     parser.add_argument('--image', required=True, help='Path to input image')
#     parser.add_argument('--exp_name', required=True, help='Experiment name (folder under experiments/)')
#     parser.add_argument('--cfg', default=None, help='Path to config.yaml (optional, defaults to experiments/<exp_name>/config.yaml)')
#     parser.add_argument('--epoch', type=int, default=None, help='Epoch number to load (defaults to last checkpoint)')
#     parser.add_argument('--save_path', default=None, help='If set, save visualized output to this path')
#     parser.add_argument('--view', action='store_true', help='Show image window with overlay')
#     parser.add_argument('--cpu', action='store_true', help='Force CPU device')
#     args = parser.parse_args()
#
#     # Setup Experiment and Config
#     exp = Experiment(args.exp_name, args, mode='test')
#     cfg_path = args.cfg if args.cfg is not None else exp.cfg_path
#     cfg = Config(cfg_path)
#     exp.set_cfg(cfg, override=False)
#
#     # Device selection
#     if args.cpu:
#         device = torch.device('cpu')
#     else:
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print('Using device:', device)
#
#     # Create runner
#     runner = Runner(cfg, exp, device, view='all' if args.view else None)
#
#     # Determine epoch
#     epoch = args.epoch if args.epoch is not None else exp.get_last_checkpoint_epoch()
#     if epoch < 0:
#         raise RuntimeError('No checkpoints found for experiment {}'.format(args.exp_name))
#
#     # Run inference
#     print(f'Running inference on image: {args.image} with checkpoint epoch {epoch}')
#     try:
#         lanes = infer_single(runner, args.image, epoch, view=args.view, save_path=args.save_path)
#         print('Decoded lanes (count={}):'.format(len(lanes)))
#         for i, lane in enumerate(lanes):
#             print(f' Lane {i}: [{len(lane)} points]')
#     except Exception as e:
#         print('Error during inference:', str(e))
#
#
# if __name__ == '__main__':
#     main()
import pickle
import random
import logging

import cv2
import torch
import numpy as np
from tqdm import tqdm, trange


class Runner:
    def __init__(self, cfg, exp, device, resume=False, view=None, deterministic=False):
        self.cfg = cfg
        self.exp = exp
        self.device = device
        self.resume = resume
        self.view = view
        self.logger = logging.getLogger(__name__)

        # Fix seeds
        torch.manual_seed(cfg['seed'])
        np.random.seed(cfg['seed'])
        random.seed(cfg['seed'])

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def train(self):
        self.exp.train_start_callback(self.cfg)
        starting_epoch = 1
        model = self.cfg.get_model()
        model = model.to(self.device)
        optimizer = self.cfg.get_optimizer(model.parameters())
        scheduler = self.cfg.get_lr_scheduler(optimizer)
        if self.resume:
            last_epoch, model, optimizer, scheduler = self.exp.load_last_train_state(model, optimizer, scheduler)
            starting_epoch = last_epoch + 1
        max_epochs = self.cfg['epochs']
        train_loader = self.get_train_dataloader()
        loss_parameters = self.cfg.get_loss_parameters()
        for epoch in trange(starting_epoch, max_epochs + 1, initial=starting_epoch - 1, total=max_epochs):
            self.exp.epoch_start_callback(epoch, max_epochs)
            model.train()
            pbar = tqdm(train_loader)
            for i, (images, labels, _) in enumerate(pbar):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = model(images, **self.cfg.get_train_parameters())
                loss, loss_dict_i = model.loss(outputs, labels, **loss_parameters)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Scheduler step (iteration based)
                scheduler.step()

                # Log
                postfix_dict = {key: float(value) for key, value in loss_dict_i.items()}
                postfix_dict['lr'] = optimizer.param_groups[0]["lr"]
                self.exp.iter_end_callback(epoch, max_epochs, i, len(train_loader), loss.item(), postfix_dict)
                postfix_dict['loss'] = loss.item()
                pbar.set_postfix(ordered_dict=postfix_dict)
            self.exp.epoch_end_callback(epoch, max_epochs, model, optimizer, scheduler)

            # Validate
            if (epoch + 1) % self.cfg['val_every'] == 0:
                self.eval(epoch, on_val=True)
        self.exp.train_end_callback()

    def eval(self, epoch, on_val=False, save_predictions=False):
        model = self.cfg.get_model()
        model_path = self.exp.get_checkpoint_path(epoch)
        self.logger.info('Loading model %s', model_path)
        model.load_state_dict(self.exp.get_epoch_model(epoch))
        model = model.to(self.device)
        model.eval()
        if on_val:
            dataloader = self.get_val_dataloader()
        else:
            dataloader = self.get_test_dataloader()
        test_parameters = self.cfg.get_test_parameters()
        predictions = []
        self.exp.eval_start_callback(self.cfg)
        with torch.no_grad():
            for idx, (images, _, _) in enumerate(tqdm(dataloader)):
                images = images.to(self.device)
                output = model(images, **test_parameters)
                prediction = model.decode(output, as_lanes=True)
                predictions.extend(prediction)
                if self.view:
                    img = (images[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    img, fp, fn = dataloader.dataset.draw_annotation(idx, img=img, pred=prediction[0])
                    if self.view == 'mistakes' and fp == 0 and fn == 0:
                        continue
                    cv2.imshow('pred', img)
                    cv2.waitKey(0)

        if save_predictions:
            with open('predictions.pkl', 'wb') as handle:
                pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.exp.eval_end_callback(dataloader.dataset.dataset, predictions, epoch)

    def get_train_dataloader(self):
        train_dataset = self.cfg.get_dataset('train')
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=self.cfg['batch_size'],
                                                   shuffle=True,
                                                   num_workers=8,
                                                   worker_init_fn=self._worker_init_fn_)
        return train_loader

    def get_test_dataloader(self):
        test_dataset = self.cfg.get_dataset('test')
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=self.cfg['batch_size'] if not self.view else 1,
                                                  shuffle=False,
                                                  num_workers=8,
                                                  worker_init_fn=self._worker_init_fn_)
        return test_loader

    def get_val_dataloader(self):
        val_dataset = self.cfg.get_dataset('val')
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=self.cfg['batch_size'],
                                                 shuffle=False,
                                                 num_workers=8,
                                                 worker_init_fn=self._worker_init_fn_)
        return val_loader

    @staticmethod
    def _worker_init_fn_(_):
        torch_seed = torch.initial_seed()
        np_seed = torch_seed // 2**32 - 1
        random.seed(torch_seed)
        np.random.seed(np_seed)