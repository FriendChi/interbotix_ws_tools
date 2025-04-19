import os
import av  # PyAV
import numpy as np
import cv2
import gc
import h5py
import time
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor, as_completed
#
#python convert3.py --dataset_path /home/ec2-user/SageMaker/AgiBotWorld-Alpha/OpenDriveLab___AgiBot-World/raw/main --save_dataset_path /home/ec2-user/SageMaker/data/Agibot
# 常量定义
JPEG_QUALITY = 50
JPEG_PARAMS = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]

def parse_args():
    parser = argparse.ArgumentParser(description="Process video and depth images into HDF5 format.")
    parser.add_argument("--dataset_path", type=str, default="./data/sample_dataset",
                        help="Path to the input dataset folder (default: ./data/sample_dataset)")
    parser.add_argument("--save_dataset_path", type=str, default="data1",
                        help="Path to save processed dataset (default: data1)")
    return parser.parse_args()


def get_frames(container):
    """
    解码视频帧并转换为 RGB 格式的 numpy 数组。
    """
    frames = []
    for frame in container.decode(video=0):
        frames.append(frame.to_rgb().to_ndarray())
        # 可选：手动释放资源，通常无需频繁调用 gc.collect()
        del frame
    return frames


def compress_frame(frame, quality=JPEG_QUALITY):
    success, encoded_frame = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        raise ValueError("Frame compression failed!")
    return encoded_frame

def compress_frames(frames, quality=JPEG_QUALITY):
    compressed_frames = []
    for frame in tqdm(frames, desc="Compressing Frames", leave=False):
        compressed_frame = compress_frame(frame, quality)
        compressed_frames.append(compressed_frame)
    return compressed_frames


def resize_frames(frames, new_width, target_height, pad_width):
    """
    对图像帧进行等比例缩放，然后在右侧填充指定宽度的黑色边缘。
    """
    resized_frames = []
    for frame in frames:
        # 等比例缩放至目标高度，并计算新的宽度
        resized = cv2.resize(frame, (new_width, target_height), interpolation=cv2.INTER_AREA)
        # 在图像右侧填充
        padded = np.pad(resized, ((0, 0), (0, pad_width), (0, 0)), mode='constant', constant_values=0)
        resized_frames.append(padded)
    return resized_frames


def pad_compressed_images(compressed_frames):
    """
    对压缩后的图像数据进行填充，使所有数组长度一致。
    """
    start_time = time.time()
    lengths = [len(frame) for frame in compressed_frames]
    target_length = max(lengths)
    
    padded_frames = []
    for frame in tqdm(compressed_frames, desc="Padding Compressed Images", leave=False):
    # for frame in compressed_frames:
        padded_frame = np.zeros(target_length, dtype='uint8')
        padded_frame[:len(frame)] = frame
        padded_frames.append(padded_frame)
    
    # print(f'Padding time: {time.time() - start_time:.2f}s')
    return padded_frames


def process_video(file_path, target_height=480, target_width=640):
    """
    处理单个视频文件：
    - 读取视频帧
    - 根据目标尺寸进行缩放与填充
    - 压缩图像帧并填充
    """
    container = av.open(file_path)
    # print(f"Processing video: {file_path}")
    
    stream = container.streams.video[0]
    original_width = stream.codec_context.width
    original_height = stream.codec_context.height
    scale_ratio = target_height / original_height
    new_width = int(original_width * scale_ratio)
    pad_width = target_width - new_width

    frames = get_frames(container)
    container.close()
    
    if pad_width != 0:
        frames = resize_frames(frames, new_width, target_height, pad_width)
    
    frames = compress_frames(frames)
    frames = pad_compressed_images(frames)
    
    # print(f"First frame shape: {frames[0].shape}, dtype: {frames[0].dtype}")
    return frames


def extract_number(filename):
    """
    从文件名中提取数字部分，假设格式为 'head_depth_000013.png'。
    """
    basename = os.path.basename(filename)
    number_str = basename.split('_')[-1].split('.')[0]
    return int(number_str)


def get_png_files(png_files):
    """
    读取并压缩 PNG 文件，转换为 JPEG 格式。
    """
    pngs = []
    for filename in tqdm(png_files, desc='Processing Depth PNGs', leave=False):
    # for filename in png_files:
        frame = cv2.imread(filename)
        if frame is not None:
            # BGR 转 RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            success, encoded_frame = cv2.imencode('.jpg', frame, JPEG_PARAMS)
            if not success:
                raise ValueError(f"Compression failed for {filename}!")
            pngs.append(encoded_frame)
        else:
            print(f"无法读取文件: {filename}")
    return pngs


def get_video_data(folder_path, data_dict):
    """
    处理文件夹下的视频和深度图，填充到数据字典中。
    """
    video_folder = os.path.join(folder_path, 'videos')
    depth_folder = os.path.join(folder_path, 'depth')

    mp4_files = [os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith(".mp4")]
    mp4_filenames = [os.path.splitext(f)[0] for f in os.listdir(video_folder) if f.endswith(".mp4")]

    png_files = [os.path.join(depth_folder, f) for f in os.listdir(depth_folder) if f.endswith(".png")]
    png_files = sorted(png_files, key=extract_number)
    
    depth_pngs = get_png_files(png_files)
    depth_pngs = pad_compressed_images(depth_pngs)
    data_dict['/observations/images/depth'] = depth_pngs

    for file_path, filename in tqdm(list(zip(mp4_files, mp4_filenames)), desc = 'mp4_files:',leave=False):
        # 文件名映射
        if filename == 'head_color':
            filename = 'cam_high'
        elif filename == 'hand_left_color':
            filename = 'cam_left_wrist'
        elif filename == 'hand_right_color':
            filename = 'cam_right_wrist'
            
        output_key = f'/observations/images/{filename}'
        # print(f"Processing {output_key}")
        frames = process_video(file_path)
        data_dict[output_key] = frames
    return data_dict


def write_data(old_folder, new_file_path, data_dict):
    """
    从旧 h5 文件读取数据，并将图像数据与读取数据写入新的 h5 文件中。
    """
    old_h5_file = os.path.join(old_folder, 'proprio_stats.h5')
    with h5py.File(old_h5_file, 'r') as f:
        data_dict['observations/qpos'] = f['state/joint/position'][()]
        data_dict['action'] = f['action/joint/position'][()]

    with h5py.File(new_file_path, 'w') as f:
        for key, value in data_dict.items():
            f.create_dataset(key, data=value)
        f.attrs['compress'] = True


def get_subfolders(directory):
    """
    获取指定目录下的所有子文件夹的路径与名称。
    """
    if 'observations' not in directory:
        directory = os.path.join(directory, 'observations')
    subfolder_paths = []
    subfolder_names = []
    for item in os.listdir(directory):
        full_path = os.path.join(directory, item)
        if os.path.isdir(full_path):
            subfolder_paths.append(full_path)
            subfolder_names.append(item)
    return subfolder_paths, subfolder_names


def process_episode(dataset_path, task_id, episode_id, save_h5_path):
    """
    处理单个 episode：
    - 读取并处理图像数据
    - 写入新 h5 文件
    """
    proprio_folder = os.path.join(dataset_path, 'proprio_stats', task_id, episode_id)
    observations_folder = os.path.join(dataset_path, 'observations', task_id, episode_id)

    data_dict = {}
    data_dict = get_video_data(observations_folder, data_dict)
    write_data(proprio_folder, save_h5_path, data_dict)
    print('finished ',task_id)
    print(save_h5_path)
    # print(time.time())

def main():
    args = parse_args()
    dataset_path = args.dataset_path
    save_dataset_path = args.save_dataset_path

    if not os.path.exists(save_dataset_path):
        os.makedirs(save_dataset_path)
        print(f"文件夹 '{save_dataset_path}' 不存在，现已创建。")
    else:
        print(f"文件夹 '{save_dataset_path}' 已存在。")

    subfolder_paths, subfolder_names = get_subfolders(dataset_path)
    tasks = []
    try:
        with ProcessPoolExecutor(max_workers=110) as executor:
            for subfolder_path, task_name in zip(subfolder_paths, subfolder_names):
                task_save_path = os.path.join(save_dataset_path, task_name)
                os.makedirs(task_save_path, exist_ok=True)
                eps_folder_paths, eps_folder_names = get_subfolders(subfolder_path)
                for idx, episode_name in enumerate(eps_folder_names):
                    if idx <1070:
                        continue
                    save_h5_path = os.path.join(task_save_path, f'episode_{idx}.hdf5')
                    
                    # 将 process_episode 函数提交给进程池
                    future = executor.submit(process_episode, dataset_path, task_name, episode_name, save_h5_path)
                    tasks.append(future)

            # 通过 as_completed 获取结果（如果需要错误处理或进度反馈）
            for future in as_completed(tasks):
                try:
                    future.result()
                except Exception as e:
                    print("Error processing an episode:", e)
    except KeyboardInterrupt:
        print("检测到键盘中断信号，正在关闭所有子进程...")
        executor.shutdown(wait=False)
        print("所有子进程已关闭。")


if __name__ == '__main__':
    main()
