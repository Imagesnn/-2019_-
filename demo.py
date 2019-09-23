# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import glob
from tools.test import *



parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')

parser.add_argument('--resume', default='', type=str, required=True,
                    metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config', default='config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
# parser.add_argument('--base_path', default='../../data/tennis', help='datasets')
# 视觉情报信息分析任务二存放车辆图片的路径
parser.add_argument('--base_path', default='../../data/car/', help='datasets')
parser.add_argument('--cpu', action='store_true', help='cpu mode')
args = parser.parse_args()

if __name__ == '__main__':

    # 设置 device, device 为cuda 或者 cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 优化训练
    torch.backends.cudnn.benchmark = True

    device = 'cpu'

    # 导入配置参数 siammask_sharp/config_davis.json
    cfg = load_config(args)

    # warnning modified by xiaoweiba
    # from custom import Custom
    from experiments.siammask_sharp.custom import Custom

    # Custom 继承自 Siammask, torch.nn.Module
    siammask = Custom(anchors=cfg['anchors'])

    # 加载训练好的模型
    if args.resume:
        assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
        siammask = load_pretrain(siammask, args.resume)

    # siamask.eval() 设置module为评估模式(evaluation mode)
    # module.to() 设置为cuda 或者 cpu
    siammask.eval().to(device)

    # 打开所以jpg 并排序 默认升序ascending order
    img_files = sorted(glob.glob(join(args.base_path, '*.jp*')))

    # 将图片读入到ims
    ims = [cv2.imread(imf) for imf in img_files]

    if len(ims) <= 0:
        raise Exception("没有读取到图片")

    height, width = ims[0].shape[0], ims[0].shape[1]
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('/home/jw/ML/SiamMask/data/demo.avi', fourcc, 15, (width, height))

    # Select ROI
    cv2.namedWindow('SiamMask', cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty('SiamMask', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    try:
        init_rect = cv2.selectROI('SiamMask', ims[0], False, False)
        # x, y ,w , h  = GetRect();
        x, y, w, h = init_rect
        # 车道标线跟踪的结果路径

    except:
        exit()

    toc = 0
    writeVideo_flag = True
    for f, im in enumerate(ims):
        tic = cv2.getTickCount()
        if f == 0:  # init
            target_pos = np.array([x + w / 2, y + h / 2])
            target_sz = np.array([w, h])
            state = siamese_init(im, target_pos, target_sz, siammask, cfg['hp'], device=device)  # init tracker
        elif f > 0:  # tracking
            state = siamese_track(state, im, mask_enable=True, refine_enable=True, device=device)  # track
            location = state['ploygon'].flatten()
            mask = state['mask'] > state['p'].seg_thr

            im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]
            cv2.polylines(im, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 2)
            cv2.imshow('SiamMask', im)

            if writeVideo_flag:
                out.write(im)

            key = cv2.waitKey(1)
            if key > 0:
                break

        toc += cv2.getTickCount() - tic
    toc /= cv2.getTickFrequency()
    fps = f / toc
    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))

    if writeVideo_flag:
        out.release()
