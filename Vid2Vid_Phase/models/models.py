
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'cycle_gan':
        assert(opt.dataset_mode == 'unaligned' or opt.dataset_mode == 'unaligned_scale')
        from .cycle_gan_model import CycleGANModel
        model = CycleGANModel()
    elif opt.model == 'recycle_gan':
        assert(opt.dataset_mode == 'unaligned_triplet' or opt.dataset_mode == 'unaligned_triplet_scale' or opt.dataset_mode == 'unaligned_scale')
        from .recycle_gan_model import RecycleGANModel
        model = RecycleGANModel()
    elif opt.model == 'reCycle_gan':
        assert(opt.dataset_mode == 'unaligned_triplet' or opt.dataset_mode == 'unaligned_triplet_scale' or opt.dataset_mode == 'unaligned_scale')
        from .reCycle_gan_model import ReCycleGANModel
        model = ReCycleGANModel()
    elif opt.model == 'unsup_single':
        assert(opt.dataset_mode == 'unaligned_scale')
        from .unsup_model_single import UnsupModel
        model = UnsupModel()
    elif opt.model == 'few_shot_cyclegan':
        assert(opt.dataset_mode == 'unaligned_scale')
        from .fewshot_cycle_gan_model import FewShotCycleGANModel
        model = FewShotCycleGANModel()
    elif opt.model == 'flow_cyclegan':
        assert(opt.dataset_mode == 'unaligned_flow')
        from .flow_cycle_gan_model import FlowCycleGANModel
        model = FlowCycleGANModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
