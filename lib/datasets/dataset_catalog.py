from lib.config import cfg


class DatasetCatalog(object):
    dataset_attrs = {
        'LinemodTest': {
            'id': 'linemod',
            'data_root': 'data/linemod/{}/JPEGImages'.format(cfg.cls_type),
            'ann_file': 'data/linemod/{}/test.json'.format(cfg.cls_type),
            'split': 'test'
        },
        'LinemodTrain': {
            'id': 'linemod',
            'data_root': 'data/linemod/{}/JPEGImages'.format(cfg.cls_type),
            'ann_file': 'data/linemod/{}/train.json'.format(cfg.cls_type),
            'split': 'train'
        },
        'LinemodOccTest': {
            'id': 'linemod',
            'data_root': 'data/occlusion_linemod/RGB-D/rgb_noseg',
            'ann_file': 'data/linemod/{}/occ.json'.format(cfg.cls_type),
            'split': 'test'
        },
        'CocoTrain': {
            'id': 'coco',
            'data_root': 'data/coco/images/train2017',
            'ann_file': 'data/coco/annotations/person_keypoints_train2017.json',
            'split': 'train'
        },
        'CocoTest': {
            'id': 'coco',
            'data_root': 'data/coco/images/val2017',
            'ann_file': 'data/coco/annotations/person_keypoints_val2017.json',
            'split': 'test'
        }
    }

    @staticmethod
    def get(name):
        attrs = DatasetCatalog.dataset_attrs[name]
        return attrs.copy()
