import os
import numpy as np
import torch
import io,re
import lmdb,pickle
from torch.utils.data import Dataset
import random
from .datasets import register_dataset
from basic_utils import load_jsonl,load_pickle
from torch.nn import functional as F

def average_to_fixed_length(visual_input,num_sample_clips ):
    num_clips = visual_input.shape[0]
    idxs = torch.arange(0, num_sample_clips+1, 1.0)/num_sample_clips*num_clips
    idxs = torch.min(torch.round(idxs).long(),torch.tensor(num_clips-1))
    new_visual_input = []
    for i in range(num_sample_clips):
        s_idx, e_idx = idxs[i].item(), idxs[i+1].item()
        if s_idx < e_idx:
            new_visual_input.append(torch.mean(visual_input[s_idx:e_idx],dim=0))
        else:
            new_visual_input.append(visual_input[s_idx])
    new_visual_input = torch.stack(new_visual_input, dim=0)
    return new_visual_input
@register_dataset("ego4d")
class Ego4dDataset(Dataset):
    def __init__(
            self,
            is_training,  # if in training mode
            split,  # split, a tuple/list allowing concat of subsets
            val_jsonl_file,  # jsonl file for validation split
            video_feat_dir,  # folder for video features
            text_feat_dir,  # folder for text features
            val_text_feat_dir,  # folder for text features of val split
            json_file,  # json file for annotations
            train_jsonl_file,  # jsonl file for annotations
            feat_stride,  # temporal stride of the feats
            num_frames,  # number of frames for each feat
            default_fps,  # default fps
            downsample_rate,  # downsample rate for feats
            max_seq_len,  # maximum sequence length during training
            input_txt_dim,  # input text feat dim
            input_vid_dim,  # input video feat dim
            num_classes,  # number of action categories
            enable_temporal_jittering,  # enable temporal jittering strategy
            fix_video_frames,#fix the video length unless it is 0
            neg_noun_text_feat_dir=None,
            object_feat_dir=None,
            classname_feat_dir=None,
            classname_feat_concat=None,
            top5_object_feat_dir=None,
            object_feat_type=None,
            lavila_caption_dir=None
    ):
        # file path
        assert os.path.exists(video_feat_dir)
        assert os.path.exists(text_feat_dir)
        assert os.path.exists(train_jsonl_file)
        assert isinstance(split, tuple) or isinstance(split, list)
        if is_training:
            self.jsonl_file = train_jsonl_file
        else:
            self.jsonl_file = val_jsonl_file

        self.json_file = json_file
        # split / training mode
        self.split = split
        self.is_training = is_training

        # features meta info
        self.input_txt_dim = input_txt_dim
        self.input_vid_dim = input_vid_dim
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.num_classes = num_classes

        self.video_visual_env = lmdb.open(video_feat_dir, readonly=True, create=False, max_readers=4096 * 8,
                                          readahead=False)
        self.video_visual_txn = self.video_visual_env.begin(buffers=True)

        if is_training:
            self.clip_textual_env = lmdb.open(text_feat_dir, readonly=True, create=False, max_readers=4096 * 8,
                                              readahead=False)
            
        else:
            self.clip_textual_env = lmdb.open(val_text_feat_dir, readonly=True, create=False, max_readers=4096 * 8,
                                              readahead=False)
        if neg_noun_text_feat_dir is not None:
                self.neg_noun_clip_textual_env = lmdb.open(neg_noun_text_feat_dir, readonly=True, create=False, max_readers=4096 * 8,
                                              readahead=False)
                self.clip_neg_noun_textual_txn = self.neg_noun_clip_textual_env.begin(buffers=True)
        else:
                self.clip_neg_noun_textual_txn = None
        if object_feat_dir is not None:

            self.object_textual_env = lmdb.open(object_feat_dir, readonly=True, create=False, max_readers=4096 * 8,
                                              readahead=False)
            self.object_textual_txn = self.object_textual_env.begin(buffers=True)
        else:
            self.object_textual_txn = None
        self.object_feat_type=object_feat_type
        if top5_object_feat_dir is not None:#top5 hs
            raise ValueError('not support error')#只保留object_feat_dir
            self.top5_object_textual_env = lmdb.open(top5_object_feat_dir, readonly=True, create=False, max_readers=4096 * 8,
                                              readahead=False)
            self.top5_object_textual_txn = self.top5_object_textual_env.begin(buffers=True)
        else:
            self.top5_object_textual_txn = None
        
        if classname_feat_dir is not None:#用于加载classname feature
            if classname_feat_concat=='token':
                self.classname_feat=load_pickle(classname_feat_dir)
                self.classname_feat=[f.cpu() for f in self.classname_feat]
            else:
                self.classname_feat=torch.load(classname_feat_dir,map_location='cpu')
            self.classname_feat_concat=classname_feat_concat
        else:
            self.classname_feat=None
        self.clip_textual_txn = self.clip_textual_env.begin(buffers=True)

        self.lavila_caption_dir=lavila_caption_dir

        # load database and select the subset
        self.data_list = self.load_data()
        self.enable_temporal_jittering = enable_temporal_jittering
        self.fix_video_frames=fix_video_frames
        # dataset specific attributes
        self.db_attributes = {
            'dataset_name': 'Ego4d',
            'nlq_tiou_thresholds': np.linspace(0.3, 0.5, 2),
            'nlq_topK': np.array([1, 5, 10, 50, 100]),
        }

        # 1/1.87*30 = 16.043
        self.fps_attributes = {
            'feat_stride': feat_stride, #16.043,
            'num_frames': num_frames, #16.043,
            'default_fps': default_fps, #30,
        }

        print("len of dataset: ", len(self.data_list))

    def get_attributes(self):
        return self.db_attributes

    def load_data(self):
        datalist = load_jsonl(self.jsonl_file)
        if self.is_training:
            random.shuffle(datalist)

        return datalist

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # directly return a (truncated) data point (so it is very fast!)
        # auto batching will be disabled in the subsequent dataloader
        # instead the model will need to decide how to batch / pre-process the data
        video_item = self.data_list[idx]
        task_name = video_item["query_type"]

        # load video features
        try:
            feats = self._get_video_feat_by_vid(video_item["video_id"])
        except:
            print(video_item["video_id"])
            exit(1)

        feat_stride = self.fps_attributes["feat_stride"] * self.downsample_rate
        
        try:
            assert len(feats) > 0
            if self.fix_video_frames==0:
                assert len(feats) <= self.max_seq_len
        except:
            print("video_item: ", video_item, len(feats))
        # T x C -> C x T
        
        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose(0, 1)))
        vid_len=feats.shape[1]
        if self.fix_video_frames!=0:
            feats=feats.transpose(0,1)
            rate=feats.shape[0]/self.fix_video_frames
            feats=average_to_fixed_length(feats,self.fix_video_frames)
            feats=feats.transpose(0,1)
            feat_stride=feat_stride*rate
            data_dict={'video_id': video_item['video_id'],
                     'feats': feats,  # C x T
                     'fps': self.fps_attributes["default_fps"],#fps不变
                     'duration': video_item['duration'],
                     'feat_stride': self.fps_attributes["feat_stride"]*rate,#特征间隔对应的帧数变了
                     'feat_num_frames': self.fps_attributes["num_frames"]+(rate-1)*self.fps_attributes["feat_stride"]#特征对应的帧数变了
                     }
        else:
        # return a data dict
            data_dict = {'video_id': video_item['video_id'],
                        'feats': feats,  # C x T
                        'fps': self.fps_attributes["default_fps"],
                        'duration': video_item['duration'],
                        'feat_stride': self.fps_attributes["feat_stride"],
                        'feat_num_frames': self.fps_attributes["num_frames"]}

        if task_name in ["narration", "nlq","nlq_narration","goal_step",'tacos']:
            # convert time stamp (in second) into temporal feature grids
            # ok to have small negative values here
            if 'timestamps' in video_item.keys():
                temp_timestamps = []
                if self.enable_temporal_jittering:
                    for item in video_item['timestamps']:
                        duration = item[1] - item[0]
                        center = (item[1] + item[0]) / 2
                        scale_ratio = random.randint(1, 10)
                        shift_number = random.uniform(-1, 1) * (scale_ratio - 1) * duration / 2
                        new_center = center - shift_number
                        temp_timestamps.append(
                            [new_center - scale_ratio * duration / 2, new_center + scale_ratio * duration / 2])
                else:
                    temp_timestamps = video_item['timestamps']

                timestamps = np.array(temp_timestamps)
                if len(timestamps.shape) == 1:
                    timestamps = timestamps.reshape(1, -1)
                segments = torch.from_numpy(
                    (timestamps * self.fps_attributes["default_fps"]) / feat_stride  # - 0.5 * self.num_frames，计算对应第几块特征
                )
                # regex = re.compile('^positive_idx_')
                # thres=20
                # if segments.shape[0]<thres:
                #     for key, value in video_item.items():
                #         if regex.match(key):
                #             frames_num=float(key.split('_')[-1])
                #             postive_segments=[]
                #             for v in value:
                #                 postive_segments.append([v,v+1])
                        
                #             postive_segments=torch.tensor(postive_segments,dtype=torch.float)*frames_num/feat_stride
                #             segments=torch.cat([segments,postive_segments],dim=0)
                #             segments=segments[:thres]

                labels = torch.zeros(len(segments), dtype=torch.int64)
                gt_label_one_hot = F.one_hot(labels, self.num_classes)
            else:
                segments, gt_label_one_hot = None, None

            data_dict.update({
                'segments': segments,  # N x 2
                'one_hot_labels': gt_label_one_hot,  # N x C
            })

            real_query_id = video_item["query_id"]
            if "narration_query" in video_item.keys():
                query = video_item['narration_query']
            else:
                query = video_item['query']
            if self.lavila_caption_dir is not None:
                file_path=os.path.join(self.lavila_caption_dir,'{}.json'.format(video_item['video_id']))
                lavila_caption=load_jsonl(file_path)
                shot_frame=[]
                for lavila_cap in lavila_caption:
                    lavila_cap_narrow=lavila_cap['narration'].lower()
                    lavila_cap_nar=lavila_cap_narrow.split(';')
                    for lavila_cap_n in lavila_cap_nar:
                        # if '#c' in lavila_cap_n and ( 'looks around' in lavila_cap_n or 'turns around' in lavila_cap_n or 'walks' in lavila_cap_n or 'moves around' in lavila_cap_n):
                        if '#c' in lavila_cap_n:
                            for word in ['looks around' , 'turns around' ]:
                            # for word in ['looks around' , 'turns around' , 'walks' , 'moves around',' rides' ,' runs',' cycles',' jogs',' jumps']:
                            # for word in ['looks around' , 'turns around' , 'walks' , 'moves around']:
                                if word in lavila_cap_n :
                        # if '#c' in lavila_cap_n and ('looks around' in lavila_cap_n or 'turns around' in lavila_cap_n):
                                    shot_frame.append(lavila_cap['frame_start_id'])
                                    break
                            if word in lavila_cap_n :
                                break
                shot_left=torch.tensor(shot_frame,dtype=torch.int)/self.fps_attributes["num_frames"]
                if shot_left.shape[0]==0 or shot_left[0]>0:
                    shot_left=torch.cat([torch.tensor([0]),shot_left],dim=0)
                shot_left=shot_left[shot_left<vid_len]#该死的，我忘了naq的特征长度是错的
                shot_left=torch.cat([shot_left,torch.tensor([vid_len])],dim=0)
                shot_right=torch.cat([shot_left[1:],torch.tensor([self.max_seq_len])],dim=0)
                # shots_len=shot_right-shot_left
                shots=torch.stack([shot_left,shot_right],dim=0)#左闭右开
                data_dict.update({
                'shots': shots,  # 2xN 
                
                })
            query_feat = self._get_query_feat_by_qid(self.clip_textual_txn,real_query_id)
            query_feat = torch.from_numpy(np.ascontiguousarray(query_feat.transpose(0, 1)))
            if self.clip_neg_noun_textual_txn is not None:
                neg_noun_query_feat = self._get_query_feat_by_qid(self.clip_neg_noun_textual_txn,real_query_id)
                neg_noun_query_feat = torch.from_numpy(np.ascontiguousarray(neg_noun_query_feat.transpose(0, 1)))
                data_dict.update({
                    'neg_noun_query_feats': neg_noun_query_feat,  # C x T

                })
            if self.object_textual_txn is not None:
                if self.classname_feat_concat=='only' and self.object_feat_type=='class-score':#narration没有物体特征,我也懒得提了，提一个视频的物体种类吧；相当于_get_class_score_feat的label相关版本
                    object_feat = self._get_object_class_feat(self.object_textual_txn,video_item['video_id'],video_item['label'],self.classname_feat)
                elif self.object_feat_type is None:#兼容旧版本
                    object_feat = self._get_object_feat(self.object_textual_txn,video_item['video_id'],video_item['label'],self.classname_feat,vid_len=vid_len)
                elif self.object_feat_type=='class-score':#好像是输入所有的类,且仅保留了大于阈值的类的最高分数，这下我不用重新搞了
                    object_feat = self._get_class_score_feat(self.object_textual_txn,video_item['video_id'],video_item['label'],self.classname_feat)
                data_dict.update({
                    'object_feats': object_feat,  #F X ([object_num,C]/None)

                })
            elif self.top5_object_textual_txn is not None:
                raise ValueError('not support error')#只保留object_feat_dir
                object_feat = self._get_top5_object_feat(self.top5_object_textual_txn,video_item['video_id'],self.classname_feat)
                data_dict.update({
                    'object_feats': object_feat,  #F X ([object_num,C]/None)

                })
            data_dict.update({
                'query_id': video_item['query_id'],
                'query': query,
                'query_feats': query_feat,  # C x T
            })   
        else:
            print("unsupported task name: ", task_name)
            exit(1)

        return data_dict
    def _get_top5_object_feat(self,object_txn,vid,classname_feat):
        
        dump = object_txn.get(vid.encode())
        if dump is None:
            print(vid)
        try:
            value = pickle.loads(dump)
        except:
            print(vid)
        object_features=[]
        for v in value:
            if v is not None:
                v=torch.from_numpy(v)
            object_features.append(v)
        return object_features
    def _get_object_class_feat(self,object_txn,vid,object_labels,classname_feat):
        
        
        class_ids=[object_label["class_id"] for object_label in object_labels]
        # class_ids=list(set(class_ids))#去重

        dump = object_txn.get(vid.encode())
        video_objs = pickle.loads(dump)

        object_features=[]
        for frame_objs in video_objs:
            if len(frame_objs) ==0:
                object_features.append(None)
                continue
            frame_obj_collect=[]
            for class_id in class_ids:
                if isinstance(class_id,int):
                    if class_id in frame_objs.keys():
                        frame_obj_collect.append(class_id)
                elif isinstance(class_id,list):
                    for class_i in class_id:
                        if class_i in frame_objs.keys():
                            frame_obj_collect.append(class_i)
                            break#只取最大的那个
                else:
                    raise ValueError('not support type')
            frame_obj_collect=list(set(frame_obj_collect))
            frame_obj_feature=classname_feat[frame_obj_collect]
            # for obj in frame_objs.keys():
            #     if obj in class_ids:
            #         frame_obj_feature.append(classname_feat[obj])#[c]
            if frame_obj_feature.shape[0]<=0:
                
                frame_obj_feature=None
            object_features.append(frame_obj_feature)
        
        return object_features#F X ([object_num,C]/None)
    def _get_class_score_feat(self,object_txn,vid,object_labels,classname_feat):
        
        dump = object_txn.get(vid.encode())

        video_objs = pickle.loads(dump)
        object_features=[]
        for frame_objs in video_objs:
            if len(frame_objs) ==0:
                object_features.append(None)
                continue
            frame_obj_feature=[]
            for obj in frame_objs.keys():
                frame_obj_feature.append(classname_feat[obj])#[c]
            frame_obj_feature=torch.stack(frame_obj_feature,dim=0)
            object_features.append(frame_obj_feature)
        
        return object_features#F X ([object_num,C]/None)
    def _get_object_feat(self,object_txn,vid,object_labels,classname_feat,vid_len):
        object_features=None
        class_ids=[object_label["class_id"] for object_label in object_labels]
        class_ids=list(set(class_ids))#去重
        for class_id in class_ids:
            # class_id=object_label["class_id"]
            key='{}_{}'.format(vid,class_id)
            dump = object_txn.get(key.encode())
            
            if dump is None:#narration没有这个object的特征
                    raise ValueError('no object feature')
            else:
                    value = pickle.loads(dump)#为什么会有空的出现？？？因为有空的注释文件：/feng_yi_sen/data/ego4d/official/v1/video/detr-anno/3fb6a472-63ad-453a-8ced-11b016d1d5f3.json
            if object_features is None:
                object_features=[[] for _ in range(len(value))]#[[] * len(value)] 这种方法创建的列表中的子列表实际上都是对同一个空列表的引用，而不是独立的列表。
            for i in range(len(value)):
                
                if value[i] is not None:
                    object_feature=torch.from_numpy(value[i])
                    if classname_feat is not None:
                        if self.classname_feat_concat=='token':
                            object_feature=classname_feat[class_id]
                        else:
                            object_feature,classname_f=torch.broadcast_tensors(object_feature,classname_feat[class_id])#[obj,c]
                            if self.classname_feat_concat=='concat':
                                
                                object_feature=torch.cat([object_feature,classname_f],dim=1)
                            elif self.classname_feat_concat=='add':
                                object_feature=object_feature+classname_f#[obj,C]
                            elif self.classname_feat_concat=='only':
                                object_feature=classname_f
                            else:
                                return NotImplementedError
                        # object_feature=torch.ones_like(object_feature)
                    object_feature=object_feature.detach()
                    object_features[i].append(object_feature)
                else:
                    object_features[i].append(value[i])
        if object_features is None:#大约有1/6的数据匹配不到标签
            return None
        torch_object_features=[]
        for frame_object_feature in object_features:#object_features是否可能为空？
            torch_frame_object_feature=[]
            for object_feature in frame_object_feature:
                if object_feature is not None:#去除None
                    
                    torch_frame_object_feature.append(object_feature)
            if len(torch_frame_object_feature)==0:
                torch_object_features.append(None)
            else:
                torch_object_features.append(torch.cat(torch_frame_object_feature,dim=0))
        return torch_object_features#F X ([object_num,C]/None)

    def _get_query_feat_by_qid(self,clip_textual_txn, qid):
        dump = clip_textual_txn.get(qid.encode())
        with io.BytesIO(dump) as reader:
            q_dump = np.load(reader, allow_pickle=True)
            try:
                q_feat = q_dump['token_features']
            except:
                q_feat = q_dump['features']

        if len(q_feat.shape) == 1:
            q_feat = np.expand_dims(q_feat, 0)

        return torch.from_numpy(q_feat)  # (Lq, D), (D, )

    def _get_video_feat_by_vid(self, vid):
        dump = self.video_visual_txn.get(vid.encode())
        with io.BytesIO(dump) as reader:
            img_dump = np.load(reader, allow_pickle=True)
            v_feat = img_dump['features'].astype(np.float32)

        return torch.from_numpy(v_feat)  # (Lv, D)
