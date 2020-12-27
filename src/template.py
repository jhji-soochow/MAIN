def set_template(args):
    # Set the templates here
    if args.template.find('jpeg') >= 0:
        args.data_train = 'DIV2K_jpeg'
        args.data_test = 'DIV2K_jpeg'
        args.epochs = 200
        args.decay = '100'

    if args.template.find('EDSR_paper') >= 0:
        args.model = 'EDSR'
        args.n_resblocks = 32
        args.n_feats = 256
        args.res_scale = 0.1

    if args.template.find('MDSR') >= 0:
        args.model = 'MDSR'
        args.patch_size = 48
        args.epochs = 650

    if args.template.find('RDN') >= 0:
        args.model = 'RDN'

    if args.template.find('DDBPN') >= 0:
        args.model = 'DDBPN'
        args.patch_size = 128
        args.scale = '4'

        args.data_test = 'Set5'

        args.batch_size = 20
        args.epochs = 1000
        args.decay = '500'
        args.gamma = 0.1
        args.weight_decay = 1e-4

        args.loss = '1*MSE'

    if args.template.find('GAN') >= 0:
        args.epochs = 200
        args.lr = 5e-5
        args.decay = '150'

    if args.template.find('RCAN') >= 0:
        args.model = 'RCAN'
        args.n_resgroups = 10
        args.n_resblocks = 20
        args.n_feats = 64
        args.chop = False

    if args.template.find('RCAN_small') >= 0:
        args.model = 'RCAN'
        args.n_resgroups = 5
        args.n_resblocks = 20
        args.n_feats = 64
        args.chop = False

    if args.template.find('RCAN_small1') >= 0:
        args.model = 'RCAN1'
        args.n_resgroups = 5
        args.n_resblocks = 20
        args.n_feats = 64
        args.chop = False

    if args.template.find('RCAN_small2') >= 0:
        args.model = 'RCAN2'
        args.n_resgroups = 5
        args.n_resblocks = 20
        args.n_feats = 64
        args.chop = False
        # args.lr = 1e-5

    if args.template.find('VDSR') >= 0:
        args.model = 'VDSR'
        args.n_resblocks = 20
        args.n_feats = 64
        args.patch_size = 41
        args.lr = 1e-1

    if args.template.find('DoGNet') >= 0:
        args.model = 'DoGNet'

    if args.template.find('DoGNet1') >= 0:
        args.model = 'DoGNet1'
        args.loops = 3
        args.gaussian_size = 5

    if args.template.find('DoGNet2') >= 0:
        args.model = 'DoGNet2'
        args.loops = 3
        args.gaussian_size = 5

    if args.template.find('DoGNet3') >= 0:
        args.model = 'DoGNet3'
        args.loops = 1
        args.gaussian_size = 7

    if args.template.find('DoGNet3_1') >= 0:
        args.model = 'DoGNet3_1'
        args.loops = 3
        args.gaussian_size = 7

    if args.template.find('DoGNet4') >= 0:
        args.model = 'DoGNet4'
        args.loops = 3
        args.gaussian_size = 7

    if args.template.find('DoGNetv2') >= 0:
        args.model = 'DoGNetv2'
        args.loops = 1
        args.gaussian_size = 7
        args.modulate_rate = 16
        args.n_resgroups = 5
        args.n_resblocks = 20
        args.n_feats = 64
        args.chop = False

    if args.template.find('DoGNetv3') >= 0:
        args.model = 'DoGNetv3'
        args.loops = 1
        args.gaussian_size = 7
        args.modulate_rate = 16
        args.n_resgroups = 5
        args.n_resblocks = 20
        args.n_feats = 64
        args.chop = False

    if args.template.find('DoGNetv4') >= 0:
        args.model = 'DoGNetv4'
        args.loops = 1
        args.gaussian_size = 7
        args.modulate_rate = 16
        args.n_resgroups = 5
        args.n_resblocks = 20
        args.n_feats = 64
        args.chop = False

    if args.template.find('DoGNetv4_1') >= 0:
        args.model = 'DoGNetv4_1'
        args.loops = 1
        args.gaussian_size = 7
        args.modulate_rate = 16
        args.n_resgroups = 5
        args.n_resblocks = 20
        args.n_feats = 64
        args.chop = False

    if args.template.find('MSRN') >= 0:
        args.model = 'MSRN'

    if args.template.find('MSRN2') >= 0:
        args.model = 'MSRN2'

    if args.template.find('MSRN3') >= 0:
        args.model = 'MSRN3'

    if args.template.find('MSRN4') >= 0:
        args.model = 'MSRN4'

    if args.template.find('DoGMSRN') >= 0:
        args.model = 'DoGMSRN'

    if args.template.find('DoGMSRNV2') >= 0:
        args.model = 'DoGMSRNV2'
        args.sigma_region = 8

    if args.template.find('DoGMSRNV3') >= 0:
        args.model = 'DoGMSRNV3'
        args.sigma_region = 8

    if args.template.find('DoGMSRNV4') >= 0:
        args.model = 'DoGMSRNV4'
        args.sigma_region = 8
    
    if args.template.find('DoGMSRNV5') >= 0:
        args.model = 'DoGMSRNV5'
        args.sigma_region = 8

    if args.template.find('DoGMSRNV6') >= 0:
        args.model = 'DoGMSRNV6'
        args.sigma_region = 8

    if args.template.find('DoGMSRNV7') >= 0:
        args.model = 'DoGMSRNV7'
        # args.sigma_region = 3


    if args.template.find('DoGMSRNV8') >= 0:
        args.model = 'DoGMSRNV8'


    if args.template.find('AIN') >=0:
        args.model = 'AIN'
        args.n_resgroups = 1
        args.n_resblocks = 7

    if args.template.find('AIN0') >=0:
        args.model = 'AIN0'
        args.n_resgroups = 1
        args.n_resblocks = 7

    if args.template.find('AIN1') >=0:
        args.model = 'AIN1'
        args.n_resgroups = 1
        args.n_resblocks = 7

    if args.template.find('AIN2') >=0:
        args.model = 'AIN2'
        args.n_resgroups = 1
        args.n_resblocks = 7

    if args.template.find('DoGAIN') >=0:
        args.model = 'DOGAIN'
        args.n_resgroups = 1
        args.n_resblocks = 7

    if args.template.find('DoGAIN1') >=0:
        args.model = 'DOGAIN1'
        args.n_resgroups = 1
        args.n_resblocks = 7

    if args.template.find('DoGAIN2') >=0:
        args.model = 'DOGAIN2'
        args.n_resgroups = 1
        args.n_resblocks = 7

    if args.template.find('DoGAIN3') >=0:
        args.model = 'DOGAIN3'
        args.n_resgroups = 1
        args.n_resblocks = 7

    if args.template.find('DoGAIN4') >=0:
        args.model = 'DOGAIN4'
        args.n_resgroups = 1
        args.n_resblocks = 7

    if args.template.find('DoGAIN5') >=0:
        args.model = 'DOGAIN5'
        args.n_resgroups = 1
        args.n_resblocks = 7
    

    if args.template.find('FSRCNN') >=0:
        args.model = 'FSRCNN'