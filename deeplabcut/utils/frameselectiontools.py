#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/DeepLabCut/DeepLabCut
Please see AUTHORS for contributors.

https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""


import math

import cv2
import numpy as np
from skimage.util import img_as_ubyte
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm


def UniformFrames(clip, numframes2pick, start, stop, Index=None):
    """Temporally uniformly sampling frames in interval (start,stop).
    Visual information of video is irrelevant for this method. This code is fast and sufficient (to extract distinct frames),
    when behavioral videos naturally covers many states.

    The variable Index allows to pass on a subindex for the frames.
    """
    print(
        "Uniformly extracting of frames from",
        round(start * clip.duration, 2),
        " seconds to",
        round(stop * clip.duration, 2),
        " seconds.",
    )
    if Index is None:
        if start == 0:
            frames2pick = np.random.choice(
                math.ceil(clip.duration * clip.fps * stop),
                size=numframes2pick,
                replace=False,
            )
        else:
            frames2pick = np.random.choice(
                range(
                    math.floor(start * clip.duration * clip.fps),
                    math.ceil(clip.duration * clip.fps * stop),
                ),
                size=numframes2pick,
                replace=False,
            )
        return frames2pick
    else:
        startindex = int(np.floor(clip.fps * clip.duration * start))
        stopindex = int(np.ceil(clip.fps * clip.duration * stop))
        Index = np.array(Index, dtype=int)
        Index = Index[(Index > startindex) * (Index < stopindex)]  # crop to range!
        if len(Index) >= numframes2pick:
            return list(np.random.permutation(Index)[:numframes2pick])
        else:
            return list(Index)


# uses openCV
def UniformFramescv2(cap, numframes2pick, start, stop, Index=None):
    """Temporally uniformly sampling frames in interval (start,stop).
    Visual information of video is irrelevant for this method. This code is fast and sufficient (to extract distinct frames),
    when behavioral videos naturally covers many states.

    The variable Index allows to pass on a subindex for the frames.
    """
    nframes = len(cap)
    print(
        "Uniformly extracting of frames from",
        round(start * nframes * 1.0 / cap.fps, 2),
        " seconds to",
        round(stop * nframes * 1.0 / cap.fps, 2),
        " seconds.",
    )

    if Index is None:
        if start == 0:
            frames2pick = np.random.choice(
                math.ceil(nframes * stop), size=numframes2pick, replace=False
            )
        else:
            frames2pick = np.random.choice(
                range(math.floor(nframes * start), math.ceil(nframes * stop)),
                size=numframes2pick,
                replace=False,
            )
        return frames2pick
    else:
        startindex = int(np.floor(nframes * start))
        stopindex = int(np.ceil(nframes * stop))
        Index = np.array(Index, dtype=int)
        Index = Index[(Index > startindex) * (Index < stopindex)]  # crop to range!
        if len(Index) >= numframes2pick:
            return list(np.random.permutation(Index)[:numframes2pick])
        else:
            return list(Index)


def KmeansbasedFrameselection(
    clip,
    numframes2pick,
    start,
    stop,
    Index=None,
    step=1,
    resizewidth=30,
    batchsize=100,
    max_iter=50,
    color=False,
):
    """This code downsamples the video to a width of resizewidth.

    The video is extracted as a numpy array, which is then clustered with kmeans, whereby each frames is treated as a vector.
    Frames from different clusters are then selected for labeling. This procedure makes sure that the frames "look different",
    i.e. different postures etc. On large videos this code is slow.

    Consider not extracting the frames from the whole video but rather set start and stop to a period around interesting behavior.

    Note: this method can return fewer images than numframes2pick."""

    print(
        "Kmeans-quantization based extracting of frames from",
        round(start * clip.duration, 2),
        " seconds to",
        round(stop * clip.duration, 2),
        " seconds.",
    )
    startindex = int(np.floor(clip.fps * clip.duration * start))
    stopindex = int(np.ceil(clip.fps * clip.duration * stop))

    if Index is None:
        Index = np.arange(startindex, stopindex, step)
    else:
        Index = np.array(Index)
        Index = Index[(Index > startindex) * (Index < stopindex)]  # crop to range!

    nframes = len(Index)
    if batchsize > nframes:
        batchsize = int(nframes / 2)

    if len(Index) >= numframes2pick:
        clipresized = clip.resize(width=resizewidth)
        ny, nx = clipresized.size
        frame0 = img_as_ubyte(clip.get_frame(0))
        if np.ndim(frame0) == 3:
            ncolors = np.shape(frame0)[2]
        else:
            ncolors = 1
        print("Extracting and downsampling...", nframes, " frames from the video.")

        if color and ncolors > 1:
            DATA = np.zeros((nframes, nx * 3, ny))
            for counter, index in tqdm(enumerate(Index)):
                image = img_as_ubyte(
                    clipresized.get_frame(index * 1.0 / clipresized.fps)
                )
                DATA[counter, :, :] = np.vstack(
                    [image[:, :, 0], image[:, :, 1], image[:, :, 2]]
                )
        else:
            DATA = np.zeros((nframes, nx, ny))
            for counter, index in tqdm(enumerate(Index)):
                if ncolors == 1:
                    DATA[counter, :, :] = img_as_ubyte(
                        clipresized.get_frame(index * 1.0 / clipresized.fps)
                    )
                else:  # attention: averages over color channels to keep size small / perhaps you want to use color information?
                    DATA[counter, :, :] = img_as_ubyte(
                        np.array(
                            np.mean(
                                clipresized.get_frame(index * 1.0 / clipresized.fps), 2
                            ),
                            dtype=np.uint8,
                        )
                    )

        print("Kmeans clustering ... (this might take a while)")
        data = DATA - DATA.mean(axis=0)
        data = data.reshape(nframes, -1)  # stacking

        kmeans = MiniBatchKMeans(
            n_clusters=numframes2pick, tol=1e-3, batch_size=batchsize, max_iter=max_iter
        )
        kmeans.fit(data)
        frames2pick = []
        for clusterid in range(numframes2pick):  # pick one frame per cluster
            clusterids = np.where(clusterid == kmeans.labels_)[0]

            numimagesofcluster = len(clusterids)
            if numimagesofcluster > 0:
                frames2pick.append(
                    Index[clusterids[np.random.randint(numimagesofcluster)]]
                )

        clipresized.close()
        del clipresized
        return list(np.array(frames2pick))
    else:
        return list(Index)


def KmeansbasedFrameselectioncv2(
    config,
    video_paths,
    numframes2pick,
    start,
    stop,
    Index=None,
    step=1,
    resizewidth=30,
    batchsize=100,
    max_iter=50,
    color=False,
):
    """This code downsamples the video to a width of resizewidth.
    The video is extracted as a numpy array, which is then clustered with kmeans, whereby each frames is treated as a vector.
    Frames from different clusters are then selected for labeling. This procedure makes sure that the frames "look different",
    i.e. different postures etc. On large videos this code is slow.

    Consider not extracting the frames from the whole video but rather set start and stop to a period around interesting behavior.

    Note: this method can return fewer images than numframes2pick.

    Attention: the flow of commands was not optimized for readability, but rather speed. This is why it might appear tedious and repetitive.
    """
    ##modify
    import os
    import sys
    import re
    import glob
    from tqdm import tqdm
    import math
    import cv2
    from sklearn.cluster import MiniBatchKMeans
    import numpy as np
    from pathlib import Path
    from skimage import io
    from skimage.util import img_as_ubyte
    from deeplabcut.utils import auxiliaryfunctions, auxfun_videos
    from deeplabcut.utils.auxfun_videos import VideoWriter
    
    project_dir = Path(config).parents[0]  # one level up from config.yaml

    print("Project directory:", project_dir)
    all_frames_path = project_dir / "all_frames.npy"
    frame_indices_path = project_dir / "frame_indices_per_video.npy"
    video_ranges_path = project_dir / "video_index_ranges.npy"


    if all_frames_path.exists() and frame_indices_path.exists() and video_ranges_path.exists():
        all_frames = np.load(all_frames_path)
        print("Loaded all_frames from file:", all_frames_path)

        frame_indices_per_video = np.load(frame_indices_path, allow_pickle=True).tolist()
        print("Loaded frame_indices_per_video from file:", frame_indices_path)

        video_index_ranges = np.load(video_ranges_path)
        print("Loaded video_index_ranges from file:", video_ranges_path)
    
    else:
        video_index_ranges = []   # need to be a  global variable
        frame_indices_per_video = []  # need to be a global variable and be remembered
        all_frames = None
        current_idx = 0
        for video in video_paths:
            print("Loading ", video)
            print("updated")
            cap = VideoWriter(video)
            nframes = len(cap)
            nx, ny = cap.dimensions
            ratio = resizewidth * 1.0 / nx
            if ratio > 1:
                raise Exception("Choice of resizewidth actually upsamples!")

            if not nframes:
                print("Video could not be opened. Skipping...")
                continue
            

            startindex = int(np.floor(nframes * start))
            stopindex = int(np.ceil(nframes * stop))

            if Index is None:
                Index = np.arange(startindex, stopindex, step)
            else:
                Index = np.array(Index)
                Index = Index[(Index > startindex) * (Index < stopindex)]  # crop to range!

            nframes = len(Index)
            if batchsize > nframes:
                batchsize = nframes // 2

            frame_indices_per_video.append(Index)
            video_index_ranges.append((current_idx, current_idx + nframes))
            current_idx += nframes
            #doesn't take into account len(index) < numframes2pick

            allocated = False
            if len(Index) >= numframes2pick:
                if (
                    np.mean(np.diff(Index)) > 1
                ):  # then non-consecutive indices are present, thus cap.set is required (which slows everything down!)
                    print("Extracting and downsampling...", nframes, " frames from the video.")
                    if color:
                        for counter, index in tqdm(enumerate(Index)):
                            cap.set_to_frame(index)  # extract a particular frame
                            frame = cap.read_frame(crop=True) #one frame into np.array
                            if frame is not None:
                                image = img_as_ubyte(
                                    cv2.resize(
                                        frame,
                                        None,
                                        fx=ratio,
                                        fy=ratio,
                                        interpolation=cv2.INTER_NEAREST,
                                    ) #plot intermedium steps to make sure it works.
                                )  # color trafo not necessary; lack thereof improves speed.
                                if (
                                    not allocated
                                ):  #'DATA' not in locals(): #allocate memory in first pass
                                    DATA = np.empty(
                                        (nframes, np.shape(image)[0], np.shape(image)[1] * 3)
                                    )
                                    allocated = True
                                DATA[counter, :, :] = np.hstack(
                                    [image[:, :, 0], image[:, :, 1], image[:, :, 2]]
                                )
                    else:
                        for counter, index in tqdm(enumerate(Index)):
                            cap.set_to_frame(index)  # extract a particular frame
                            frame = cap.read_frame(crop=True)
                            if frame is not None:
                                image = img_as_ubyte(
                                    cv2.resize(
                                        frame,
                                        None,
                                        fx=ratio,
                                        fy=ratio,
                                        interpolation=cv2.INTER_NEAREST,
                                    )
                                )  # color trafo not necessary; lack thereof improves speed.
                                if (
                                    not allocated
                                ):  #'DATA' not in locals(): #allocate memory in first pass
                                    DATA = np.empty(
                                        (nframes, np.shape(image)[0], np.shape(image)[1])
                                    )
                                    allocated = True
                                DATA[counter, :, :] = np.mean(image, 2)
                else:
                    print("Extracting and downsampling...", nframes, " frames from the video.")
                    if color:
                        for counter, index in tqdm(enumerate(Index)):
                            frame = cap.read_frame(crop=True)
                            if frame is not None:
                                image = img_as_ubyte(
                                    cv2.resize(
                                        frame,
                                        None,
                                        fx=ratio,
                                        fy=ratio,
                                        interpolation=cv2.INTER_NEAREST,
                                    )
                                )  # color trafo not necessary; lack thereof improves speed.
                                if (
                                    not allocated
                                ):  #'DATA' not in locals(): #allocate memory in first pass
                                    DATA = np.empty(
                                        (nframes, np.shape(image)[0], np.shape(image)[1] * 3)
                                    )
                                    allocated = True
                                DATA[counter, :, :] = np.hstack(
                                    [image[:, :, 0], image[:, :, 1], image[:, :, 2]]
                                )
                    else:
                        for counter, index in tqdm(enumerate(Index)):
                            frame = cap.read_frame(crop=True)
                            if frame is not None:
                                image = img_as_ubyte(
                                    cv2.resize(
                                        frame,
                                        None,
                                        fx=ratio,
                                        fy=ratio,
                                        interpolation=cv2.INTER_NEAREST,
                                    )
                                )  # color trafo not necessary; lack thereof improves speed.
                                if (
                                    not allocated
                                ):  #'DATA' not in locals(): #allocate memory in first pass
                                    DATA = np.empty(
                                        (nframes, np.shape(image)[0], np.shape(image)[1])
                                    )
                                    allocated = True
                                DATA[counter, :, :] = np.mean(image, 2)

            data = DATA - DATA.mean(axis=0)
            data = data.reshape(nframes, -1)  # stacking
            print(data.shape)
            
            if all_frames is None:
                all_frames = data  # First video, set all_frames
            else:
                all_frames = np.concatenate((all_frames, data), axis=0)  # Append frames
                print(all_frames.shape)
        
            Index = None
        #all_frames : (total_frames_from_all_videos, flattened_frame_dimension)
        #no color : flattened_frame_dimension = height * width
        # Save all_frames
        np.save(all_frames_path, all_frames)
        print(f"Saved all_frames to {all_frames_path}")

        # Save video_index_ranges
        np.save(video_ranges_path, video_index_ranges)
        print(f"Saved video_index_ranges to {video_ranges_path}")

        # Ensure frame_indices_per_video is properly formatted before saving
        frame_indices_per_video = [np.array(index_array) for index_array in frame_indices_per_video]
        np.save(frame_indices_path, np.array(frame_indices_per_video, dtype=object), allow_pickle=True)
        print(f"Saved frame_indices_per_video to {frame_indices_path}")
        
    if all_frames is None:
        print("No frames extracted.")
        return
    
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import umap
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.cluster import MiniBatchKMeans

    final_k = 30
    kmeans = MiniBatchKMeans(
        n_clusters=final_k, tol=1e-3, batch_size=batchsize, max_iter=max_iter
    )
    kmeans.fit(all_frames)
    labels = kmeans.labels_

    # Dimensionality reduction
    tsne_embedding = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42).fit_transform(all_frames)

    frames2pick = []
    print("Saving")
    for clusterid in range(final_k):
        clusterids = np.where(clusterid == kmeans.labels_)[0]
        if len(clusterids) > 0:
            selected = np.random.choice(clusterids)
            frames2pick.append((clusterid, selected))  # tuple: (cluster, frame index)
    if not frames2pick:
        print("Frame selection failed.")
        return

    print("Saving selected frames...")
    
    # Map each global index to a specific video + frame
    is_valid = []
    for clusterid, idx in frames2pick:
        for vid, (start, end) in enumerate(video_index_ranges):
            if start <= idx < end:
                local_index = idx - start
                frame_number = frame_indices_per_video[vid][local_index]
                video_name = Path(video_paths[vid]).stem
                print(f"Saving frame {frame_number} from cluster {clusterid} and video '{video_name}'")
                video_path = video_paths[vid]
                cap = cv2.VideoCapture(video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()

                if ret and frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = img_as_ubyte(frame)
                    indexlength = int(np.ceil(np.log10(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
                    
                    output_path = Path(config).parents[0] / "labeled-data" / Path(video_path).stem
                    output_path.mkdir(parents=True, exist_ok=True)

                    img_name = (
                        str(output_path)
                        + "/img"
                        + str(frame_number).zfill(indexlength)
                        + ".png"
                    )
                    io.imsave(img_name, image)
                    is_valid.append(True)
                else:
                    print("Frame", frame_number, "not found!")
                    is_valid.append(False)

                cap.release()

    if not any(is_valid):
        print("All selected frames were invalid or could not be saved.")
        return []
    def plot_embedding(embedding, title, labeled_points=None):
        plt.figure(figsize=(6, 6))
        sns.scatterplot(
            x=embedding[:, 0], y=embedding[:, 1], hue=labels,
            palette="Set1", s=10, alpha=0.8, legend=False
        )
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
        plt.text(
            min(embedding[:, 0]), max(embedding[:, 1]),
            f"inertia: {kmeans.inertia_:.6f}", fontsize=10
        )

        # Add cluster ID labels on selected points
        if labeled_points:
            for clusterid, point_idx in labeled_points:
                x, y = embedding[point_idx]
                plt.text(x, y, str(clusterid), fontsize=9, weight='bold', color='black', ha='center')

        plt.show()


    # Plot UMAP and t-SNE
    plot_embedding(tsne_embedding, "t-SNE Frame Selection Visualization", labeled_points=frames2pick)

    return frames2pick

    '''nframes = len(cap)
    nx, ny = cap.dimensions
    ratio = resizewidth * 1.0 / nx
    if ratio > 1:
        raise Exception("Choice of resizewidth actually upsamples!")

    print(
        "Kmeans-quantization based extracting of frames from",
        round(start * nframes * 1.0 / cap.fps, 2),
        " seconds to",
        round(stop * nframes * 1.0 / cap.fps, 2),
        " seconds.",
    )
    startindex = int(np.floor(nframes * start))
    stopindex = int(np.ceil(nframes * stop))

    if Index is None:
        Index = np.arange(startindex, stopindex, step)
    else:
        Index = np.array(Index)
        Index = Index[(Index > startindex) * (Index < stopindex)]  # crop to range!

    nframes = len(Index)
    if batchsize > nframes:
        batchsize = nframes // 2

    ny_ = np.round(ny * ratio).astype(int)
    nx_ = np.round(nx * ratio).astype(int)
    DATA = np.empty((nframes, ny_, nx_ * 3 if color else nx_))
    if len(Index) >= numframes2pick:
        if (
            np.mean(np.diff(Index)) > 1
        ):  # then non-consecutive indices are present, thus cap.set is required (which slows everything down!)
            print("Extracting and downsampling...", nframes, " frames from the video.")
            if color:
                for counter, index in tqdm(enumerate(Index)):
                    cap.set_to_frame(index)  # extract a particular frame
                    frame = cap.read_frame(crop=True)
                    if frame is not None:
                        image = img_as_ubyte(
                            cv2.resize(
                                frame,
                                None,
                                fx=ratio,
                                fy=ratio,
                                interpolation=cv2.INTER_NEAREST,
                            )
                        )  # color trafo not necessary; lack thereof improves speed.
                        DATA[counter, :, :] = np.hstack(
                            [image[:, :, 0], image[:, :, 1], image[:, :, 2]]
                        )
            else:
                for counter, index in tqdm(enumerate(Index)):
                    cap.set_to_frame(index)  # extract a particular frame
                    frame = cap.read_frame(crop=True)
                    if frame is not None:
                        image = img_as_ubyte(
                            cv2.resize(
                                frame,
                                None,
                                fx=ratio,
                                fy=ratio,
                                interpolation=cv2.INTER_NEAREST,
                            )
                        )  # color trafo not necessary; lack thereof improves speed.
                        DATA[counter, :, :] = np.mean(image, 2)
        else:
            print("Extracting and downsampling...", nframes, " frames from the video.")
            if color:
                for counter, index in tqdm(enumerate(Index)):
                    frame = cap.read_frame(crop=True)
                    if frame is not None:
                        image = img_as_ubyte(
                            cv2.resize(
                                frame,
                                None,
                                fx=ratio,
                                fy=ratio,
                                interpolation=cv2.INTER_NEAREST,
                            )
                        )  # color trafo not necessary; lack thereof improves speed.
                        DATA[counter, :, :] = np.hstack(
                            [image[:, :, 0], image[:, :, 1], image[:, :, 2]]
                        )
            else:
                for counter, index in tqdm(enumerate(Index)):
                    frame = cap.read_frame(crop=True)
                    if frame is not None:
                        image = img_as_ubyte(
                            cv2.resize(
                                frame,
                                None,
                                fx=ratio,
                                fy=ratio,
                                interpolation=cv2.INTER_NEAREST,
                            )
                        )  # color trafo not necessary; lack thereof improves speed.
                        DATA[counter, :, :] = np.mean(image, 2)

        print("Kmeans clustering ... (this might take a while)")
        data = DATA - DATA.mean(axis=0)
        data = data.reshape(nframes, -1)  # stacking

        kmeans = MiniBatchKMeans(
            n_clusters=numframes2pick, tol=1e-3, batch_size=batchsize, max_iter=max_iter
        )
        kmeans.fit(data)
        frames2pick = []
        for clusterid in range(numframes2pick):  # pick one frame per cluster
            clusterids = np.where(clusterid == kmeans.labels_)[0]

            numimagesofcluster = len(clusterids)
            if numimagesofcluster > 0:
                frames2pick.append(
                    Index[clusterids[np.random.randint(numimagesofcluster)]]
                )
        # cap.release() >> still used in frame_extraction!
        return list(np.array(frames2pick))
    else:
        return list(Index)'''
