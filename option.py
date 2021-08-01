import argparse

parser = argparse.ArgumentParser()

# data parameters
parser.add_argument('--data_path', type=str, default='./data/ABC/')
parser.add_argument('--dataset', type=str, default='ABC')
parser.add_argument('--train_dataset',
                    type=str,
                    default='train_data.txt',
                    help='file name for the list of object names for training')
parser.add_argument('--test_dataset',
                    type=str,
                    default='test_data.txt',
                    help='file name for the list of object names for testing')
parser.add_argument('--checkpoint_path',
                    default=None,
                    help='Model checkpoint path [default: None]')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--vis',
                    action='store_true',
                    help='whether do the visualization')
parser.add_argument('--vis_dir',
                    type=str,
                    default=None,
                    help='visualization directory')
parser.add_argument('--eval', 
                    action='store_true', 
                    help='evaluate iou error')
parser.add_argument('--debug',
                    action='store_true',
                    help='whether switch to debug module')
parser.add_argument('--MEAN_SHIFT_STEP',
                    type=int,
                    default=5,
                    help='whether switch to debug module')
parser.add_argument('--log_dir',
                    default='./log/test',
                    help='Dump dir to save model checkpoint [default: log]')

# training parameters
parser.add_argument('--max_epoch',
                    type=int,
                    default=1500,
                    help='Epoch to run [default: 180]')
parser.add_argument('--learning_rate',
                    type=float,
                    default=1e-3,
                    help='Initial learning rate [default: 0.001]')
parser.add_argument('--optimizer',
                    type=str,
                    default='adam',
                    help='[adam, sgd]')
parser.add_argument('--weight_decay',
                    type=float,
                    default=0,
                    help='Optimization L2 weight decay [default: 0]')
parser.add_argument('--momentum',
                    type=float,
                    default=0.9,
                    help='Optimization L2 weight decay [default: 0]')
parser.add_argument('--bn_decay_step',
                    type=int,
                    default=20,
                    help='Period of BN decay (in epochs) [default: 20]')
parser.add_argument('--bn_decay_rate',
                    type=float,
                    default=0.5,
                    help='Decay rate for BN decay [default: 0.5]')
parser.add_argument('--lr_decay_steps',
                    default='40',
                    help='When to decay the learning rate (in epochs) [default: 80,120,160]')
parser.add_argument('--lr_decay_rates',
                    default='0.1,0.1,0.1',
                    help='Decay rates for lr decay [default: 0.1,0.1,0.1]')
parser.add_argument('--lr_decay_rate',
                    type=float,
                    default=0.1,
                    help='Decay rates for lr decay')
parser.add_argument('--loss_class',
                    type=str,
                    default='frp',
                    help='loss functions; f:embedding loss; r:primitive loss;\
                          p:parameter loss, n:normal loss')
parser.add_argument('--val_skip',
                    type=int,
                    default=100,
                    help='only test sub dataset')
parser.add_argument('--train_skip',
                    type=int,
                    default=1,
                    help='only train sub dataset')
parser.add_argument('--train_fold',
                    type=int,
                    default=1)
parser.add_argument('--eval_interval',
                    type=int,
                    default=3,
                    help='evaluation interval')
parser.add_argument('--save_interval',
                    type=int,
                    default=6,
                    help='save specific checkpoint interval')
parser.add_argument('--augment',
                    type=int,
                    default=0,
                    help='whether do data augment')
parser.add_argument('--if_normal_noise',
                    type=int,
                    default=0,
                    help='whether do normal noise')
parser.add_argument('--optimize',
                    type=int,
                    default=0,
                    help='0: optimize feat loss; 1:optimize miou')


# model parameters
parser.add_argument('--gpu',
                    type=str,
                    default='0',
                    help='gpu number')
parser.add_argument('--not_load_model',
                    action='store_true',
                    help='whether load model from checkpoint')
parser.add_argument('--model_dict',
                    type=str,
                    default='models.dgcnn',
                    help='model file name')
parser.add_argument('--sigma',
                    type=float,
                    default=0.8,
                    help='affinity matrix hyper paramter')
parser.add_argument('--normal_sigma',
                    type=float,
                    default=0.1,
                    help='normal difference affinity matrix hyper paramter')
parser.add_argument('--out_dim',
                    type=int,
                    default=128,
                    help='output feature dimension')
parser.add_argument('--type_weight',
                    type=float,
                    default=1.0,
                    help='type loss weight')
parser.add_argument('--param_weight',
                    type=float,
                    default=0.1,
                    help='parameter loss weight')
parser.add_argument('--normal_weight',
                    type=float,
                    default=1.0,
                    help='normal loss weight')

parser.add_argument('--input_normal',
                    type=int,
                    default=0,
                    help='whether input normal')
parser.add_argument('--edge_knn',
                    type=int,
                    default=50,
                    help='k nearest neighbor of normal')

parser.add_argument('--feat_ent_weight',
                    type=float,
                    default=1.70,
                    help='network feature entropy weight')
parser.add_argument('--dis_ent_weight',
                    type=float,
                    default=1.10,
                    help='primitive distance entropy weight')
parser.add_argument('--edge_ent_weight',
                    type=float,
                    default=1.23,
                    help='edge boundary entropy weight')
parser.add_argument('--topK',
                    type=int,
                    default=10,
                    help='the number of eigenvectors used')
parser.add_argument('--edge_topK',
                    type=int,
                    default=12,
                    help='the number of eigenvectors edge feature used')
parser.add_argument('--bandwidth',
                    type=float,
                    default=0.85,
                    help='kernl bandwidth')
parser.add_argument('--backbone', 
                    type=str, 
                    default='DGCNN')

def build_option():
    FLAGS = parser.parse_args()
    return FLAGS
