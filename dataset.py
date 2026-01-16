from datasets.Brain4cars import Brain4cars_Inside, Brain4cars_Outside, Brain4cars_Unit
import pdb

def get_training_set(opt, spatial_transform_invideo, spatial_transform_outvideo,horizontal_flip, temporal_transform,target_transform):
    
    assert opt.dataset in ['Brain4cars_Inside', 'Brain4cars_Outside','Brain4cars_Unit']

    if opt.dataset == 'Brain4cars_Inside':
        training_data = Brain4cars_Inside(
                opt.video_path,
                opt.annotation_path,
                'training',
                opt.n_fold,
                opt.end_second,
                1,
                spatial_transform_invideo=spatial_transform_invideo,
                spatial_transform_outvideo=None,
                horizontal_flip=horizontal_flip,
                temporal_transform=temporal_transform,
                target_transform=target_transform)
    elif opt.dataset == 'Brain4cars_Outside':
        training_data = Brain4cars_Outside(
            opt.video_path,
            opt.annotation_path,
            'training',
            opt.n_fold,
            opt.end_second,
            1,         
            spatial_transform_invideo=None,
            spatial_transform_outvideo=spatial_transform_outvideo,
            horizontal_flip=horizontal_flip,
            target_transform=target_transform,
            temporal_transform=temporal_transform)
    elif opt.dataset == 'Brain4cars_Unit':
        training_data = Brain4cars_Unit(
            opt.video_path,
            opt.annotation_path,
            'training',
            opt.n_fold,
            opt.end_second,
            1,         
            spatial_transform_invideo=spatial_transform_invideo,
            spatial_transform_outvideo=spatial_transform_outvideo,
            horizontal_flip=horizontal_flip,
            target_transform=target_transform,
            temporal_transform=temporal_transform)
    return training_data


def get_validation_set(opt,spatial_transform_invideo, spatial_transform_outvideo, temporal_transform,
                       target_transform):
    assert opt.dataset in ['Brain4cars_Inside', 'Brain4cars_Outside','Brain4cars_Unit']

    if opt.dataset == 'Brain4cars_Inside':
        validation_data = Brain4cars_Inside(
                opt.video_path,
                opt.annotation_path,
                'validation',
                opt.n_fold,
                opt.end_second,
                opt.n_val_samples,
                spatial_transform_invideo,
                None,
                None,
                temporal_transform,
                target_transform,
                sample_duration=opt.sample_duration)

    elif opt.dataset == 'Brain4cars_Outside':
        validation_data = Brain4cars_Outside(
            opt.video_path,
            opt.annotation_path,
            'validation',
            opt.n_fold,
            opt.end_second,
            opt.n_val_samples,
            spatial_transform_invideo=None,
            spatial_transform_outvideo=spatial_transform_outvideo,
            horizontal_flip=None,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration)
    
    elif opt.dataset == 'Brain4cars_Unit':
        
        validation_data = Brain4cars_Unit(
            opt.video_path,
            opt.annotation_path,
            'validation',
            opt.n_fold,
            opt.end_second,
            opt.n_val_samples,
            spatial_transform_invideo=spatial_transform_invideo,
            spatial_transform_outvideo=spatial_transform_outvideo,
            horizontal_flip=None,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration)

    return validation_data
