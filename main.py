import argparse
import itertools
import random
import json


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='objects365 toolkits')
    parser.add_argument('annotation', type=str, help='annotation file',
                        default='/mnt/data/Objects365/zhiyuan_objv2_train.json')
    parser.add_argument('output-dir', type=str, help='output dir',
                        default='/mnt/data/Objects365/parts/')
    parser.add_argument('--seed', type=int, help='random seed',
                        default=17)
    parser.add_argument('--num-part', type=int, help='num of parts',
                        default=8)
    parser.add_argument('--max-obj-per-img', type=int, help='max number of objs per image',
                        default=20)
    parser.add_argument('--min-obj-per-img', type=int, help='min number of objs per image',
                        default=1)
    parser.add_argument('--use-crowd', action='store_true', help='use crowd annotations')
    args = parser.parse_args()
    random.seed(args.seed)

    # filter
    with open(args.annotation, "r") as f:
        anns = json.load(f)

    image_list = anns['images']

    img2anns = dict([(img['id'], []) for img in image_list])

    for ann in anns['annotations']:
        if args.use_crowd or not ann['iscrowd']:
            img2anns[ann['image_id']].append(ann)

    final_image_ids = []
    for key, value in img2anns.items():
        if args.min_obj_per_img <= len(value) <= args.max_obj_per_img:
            final_image_ids.append(value[0]['image_id'])

    image_dict = dict([(img['id'], img) for img in image_list])

    final_image_list = []
    final_ann_list = []
    for img_idx in final_image_ids:
        final_image_list.append(image_dict[img_idx])
        final_ann_list.append(img2anns[img_idx])

    random.shuffle(image_list)

    # subsample
    part_size = (len(final_image_list) + args.num_part - 1) // args.num_part
    image_part_list = [final_image_list[part_size * i: part_size * (i + 1)] for i in range(args.num_part)]
    ann_part_list = [[] for i in range(args.num_part)]
    img2part = dict()
    for i in range(args.num_part):
        for j in range(i * part_size, min((i + 1) * part_size, len(final_image_list))):
            img2part[final_image_list[j]['id']] = i

    for img in final_image_list:
        image_part_list[img2part[img['id']]].append(img)
        ann_part_list[img2part[img['id']]].extend(img2anns[img['id']])

    # save the results
    for idx, (image_part, ann_part) in enumerate(zip(image_part_list, ann_part_list)):
        part_ann = dict()
        for key, value in anns.items():
            if key == 'images':
                part_ann['images'] = image_part
            elif key == 'annotations':
                part_ann['annotations'] = ann_part
            else:
                part_ann[key] = value

        with open("{}/objects365_train_part{}.json".format(args.output_dir, idx), "w") as f:
            json.dump(part_ann, f)
