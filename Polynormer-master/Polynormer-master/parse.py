from model import Graph2MapClassifier, create_vision_model

def parse_method(args, n, c, d, device):
    use_node_feat = not getattr(args, 'no_node_features', False)
    backbone = getattr(args, 'backbone', 'resnet18')
    pretrained = getattr(args, 'pretrained', False)
    model = create_vision_model(
        num_classes=c,
        node_feat_dim=d,
        backbone_name=backbone,
        pretrained=pretrained,
        use_node_features=use_node_feat,
        hidden_dim=args.hidden_channels,
        dropout=args.dropout
    ).to(device)
    return model


def parser_add_main_args(parser):
    # dataset and evaluation
    parser.add_argument('--dataset', type=str, default='amazon-photo')
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--epochs', type=int, default=100,
                        help='total training epochs for vision model')
    parser.add_argument('--local_epochs', type=int, default=50,
                        help='kept for compatibility with original scripts')
    parser.add_argument('--global_epochs', type=int, default=50,
                        help='kept for compatibility with original scripts')
    parser.add_argument('--runs', type=int, default=1,
                        help='number of distinct runs')
    parser.add_argument('--metric', type=str, default='acc', choices=['acc', 'rocauc'],
                        help='evaluation metric')

    # Graph2Map Vision Model args
    parser.add_argument('--method', type=str, default='graph2map')
    parser.add_argument('--backbone', type=str, default='resnet18',
                        help='vision backbone from timm (e.g. resnet18, resnet34, convnext_tiny, efficientnet_b0)')
    parser.add_argument('--pretrained', action='store_true',
                        help='whether to use pretrained vision backbone weights')
    parser.add_argument('--no_node_features', action='store_true',
                        help='disable fusing raw target node features (pure vision mode)')
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.3)

    # Ego-Map Rasterization & Caching
    parser.add_argument('--resolution', type=int, default=128,
                        help='canvas resolution for 2D Ego-Map (default 128x128)')
    parser.add_argument('--num_hops', type=int, default=2,
                        help='number of hops for computational ego-neighborhood')
    parser.add_argument('--max_nodes', type=int, default=128,
                        help='max bounded nodes in ego-subgraph')
    parser.add_argument('--layout_method', type=str, default='concentric',
                        choices=['concentric', 'pinned_spring'],
                        help='layout algorithm: concentric (fast polar rings) or pinned_spring')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size for vision training')
    parser.add_argument('--eval_batch_size', type=int, default=256,
                        help='batch size for evaluation inference')

    # training
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # display and utility
    parser.add_argument('--display_step', type=int,
                        default=1, help='how often to print')
    parser.add_argument('--save_model', action='store_true', help='whether to save model')
    parser.add_argument('--model_dir', type=str, default='./model/', help='where to save model')
    parser.add_argument('--save_result', action='store_true', help='whether to save result')


