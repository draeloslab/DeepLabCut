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
import os
import sys
import re
import glob
import numpy as np
from pathlib import Path
from skimage import io
from skimage.util import img_as_ubyte
from deeplabcut.utils import auxiliaryfunctions


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

from deeplabcut.utils import auxiliaryfunctions
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

##TODO: modify argument, cap replaces with allframes, take in arrays instead,
##finish the step in frameextraction functoin. 

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


    '''import numpy as np
    import cv2
    from tqdm import tqdm
    from skimage.util import img_as_ubyte
    from sklearn.cluster import MiniBatchKMeans
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

    all_frames = None

    for video in video_paths:
        print("Loading", video)
        print("fixed?")
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            print("Video could not be opened. Skipping...")
            continue

        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        resizeHeight = 30

        startindex = int(np.floor(nframes * start))
        stopindex = int(np.ceil(nframes * stop))

## index something wrong
        if Index is None:
            Index = np.arange(startindex, stopindex, step)
        else:
            Index = np.array(Index)
            Index = Index[(Index > startindex) & (Index < stopindex)]

        nframes = len(Index)
        if batchsize > nframes:
            batchsize = max(1, nframes // 2)

        allocated = False
        if len(Index) >= numframes2pick:
            print("Extracting and downsampling...", nframes, "frames from the video.")

            if color:
                for counter, index in tqdm(enumerate(Index)):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                    success, frame = cap.read()
                    if success and frame is not None:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image = img_as_ubyte(
                            cv2.resize(frame, (resizewidth, resizeHeight), interpolation=cv2.INTER_NEAREST)
                        )
                        if not allocated:
                            DATA = np.empty((nframes, image.shape[0], image.shape[1] * 3))
                            allocated = True
                        DATA[counter, :, :] = np.hstack([image[:, :, 0], image[:, :, 1], image[:, :, 2]])
            else:
                for counter, index in tqdm(enumerate(Index)):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                    success, frame = cap.read()
                    if success and frame is not None:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image = img_as_ubyte(
                            cv2.resize(frame, (resizewidth, resizeHeight), interpolation=cv2.INTER_NEAREST)
                        )
                        if not allocated:
                            DATA = np.empty((nframes, image.shape[0], image.shape[1]))
                            allocated = True
                        DATA[counter, :, :] = np.mean(image, axis=2)

        cap.release()
        del cap
        Index = None

        data = DATA - DATA.mean(axis=0)
        data = data.reshape(nframes, -1)

        if all_frames is None:
            all_frames = data
        else:
            all_frames = np.concatenate((all_frames, data), axis=0)

    if all_frames is None:
        print("No frames extracted.")
        return

    print("Custom DeepLabCut modification loaded!")

    # KMeans Visualization
    clusterer = MiniBatchKMeans(random_state=10)
    visualizer = KElbowVisualizer(clusterer, k=(2, 15), metric='calinski_harabasz', locate_elbow=True, timings=False)
    visualizer.fit(all_frames)
    visualizer.show()
    optimal_k = visualizer.elbow_value_
    print(f"Optimal number of clusters determined by Elbow Method: {optimal_k}")

    silhouette_scores = {}
    for n_clusters in range(2, 16):
        clusterer = MiniBatchKMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(all_frames)
        silhouette_avg = silhouette_score(all_frames, cluster_labels)
        silhouette_scores[n_clusters] = silhouette_avg
        print(f"For n_clusters = {n_clusters}, The average silhouette_score is: {silhouette_avg}")
        visualizer = SilhouetteVisualizer(clusterer, colors='yellowbrick')
        visualizer.fit(all_frames)
        visualizer.show()

    optimal_k_silhouette = max(silhouette_scores, key=silhouette_scores.get)
    print(f"Optimal number of clusters determined by Silhouette Score: {optimal_k_silhouette}")

    final_k = optimal_k if optimal_k == optimal_k_silhouette else optimal_k_silhouette

    print("KMeans clustering ... (this might take a while)")
    kmeans = MiniBatchKMeans(n_clusters=final_k, tol=1e-3, batch_size=batchsize, max_iter=max_iter)
    kmeans.fit(all_frames)

    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data)

    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=data_pca[:, 0], y=data_pca[:, 1], hue=kmeans.labels_, palette="Set1", s=10, alpha=0.8, legend=False)
    cluster_centers = pca.transform(kmeans.cluster_centers_)
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c="black", s=100, edgecolors="white", marker="o")
    plt.text(min(data_pca[:, 0]), max(data_pca[:, 1]), f"inertia: {kmeans.inertia_:.6f}", fontsize=10)
    plt.title("K-Means Frame Selection Visualization")
    plt.xticks([])
    plt.yticks([])
    plt.show()

    frames2pick = []
    for clusterid in range(final_k):
        clusterids = np.where(clusterid == kmeans.labels_)[0]
        if len(clusterids) > 0:
            frames2pick.append(Index[np.random.choice(clusterids)])

    return list(frames2pick)'''

    ##get access to 20 videos ask jack.
        ##what level of downsampling for live, smaller image, what jack is using, train similar resolution, 
        ##resize the image visulization
        ##check nframes, own script to uses opencv

    
    '''
    #delete old files py if needed
    files_to_delete = [
        "all_frames.npy",
        "video_index_ranges.npy",
        "frame_indices_per_video.npy",
    ]

    for filename in files_to_delete:
        if os.path.exists(filename):
            os.remove(filename)
            print(f"Deleted {filename}")
        else:
            print(f"{filename} does not exist.")'''
            
    if os.path.exists("frame_indices_per_video.npy") and os.path.exists("video_index_ranges.npy") and os.path.exists("all_frames.npy"):
        all_frames = np.load("all_frames.npy")
        print("Loaded all_frames from file.")
        frame_indices_per_video = np.load("frame_indices_per_video.npy", allow_pickle=True).tolist()
        print("Loaded frame_indices_per_video from file.")
        video_index_ranges = np.load("video_index_ranges.npy")
        print("Loaded video_index_ranges from file.")
        
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
            ##dimension is different for different videos, shape[1]is different
            #fixed it by using the same width and height for different videos, don't know the performance tho
            if all_frames is None:
                all_frames = data  # First video, set all_frames
            else:
                all_frames = np.concatenate((all_frames, data), axis=0)  # Append frames
                print(all_frames.shape)
        
            Index = None
        #all_frames : (total_frames_from_all_videos, flattened_frame_dimension)
        #no color : flattened_frame_dimension = height * width
        np.save("all_frames.npy", all_frames)
        print("Saved all_frames to all_frames.npy")
        np.save("video_index_ranges.npy", video_index_ranges)
        print("Saved video_index_ranges to video_index_ranges.npy")
        # Ensure each item is a NumPy array
        frame_indices_per_video = [np.array(index_array) for index_array in frame_indices_per_video]
        # Save with enforced object dtype
        np.save("frame_indices_per_video.npy", np.array(frame_indices_per_video, dtype=object), allow_pickle=True)
        print("Saved frame_indices_per_video to frame_indices_per_video.npy")
        
    if all_frames is None:
        print("No frames extracted.")
        return
        #added begin

    print(all_frames.shape)
        # Visualization of K-Means Clustering Results
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.decomposition import PCA
         
    import numpy as np
    import cv2
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    from sklearn.metrics import davies_bouldin_score
    #DBI method
    '''k_values = range(2, 50)
    dbi_scores = []

    print("Evaluating clustering quality with Davies-Bouldin Index...")
    for k in k_values:
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=10, batch_size=batchsize, max_iter=max_iter)
        kmeans.fit(all_frames)
        labels = kmeans.labels_

        dbi = davies_bouldin_score(all_frames, labels)
        dbi_scores.append(dbi)
        print(f"k = {k}, DBI = {dbi:.4f}")

    # Plot DBI vs k
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, dbi_scores, marker='o', linestyle='-')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Davies-Bouldin Index")
    plt.title("Optimal k by Davies-Bouldin Index")
    plt.grid(True)
    plt.show()

    # Find the optimal k (minimum DBI)
    optimal_k = k_values[dbi_scores.index(min(dbi_scores))]
    print(f"Optimal number of clusters based on DBI: {optimal_k}")'''

    '''from sklearn.metrics import calinski_harabasz_score

# Define your range of cluster counts
    k_values = range(2, 40, 2)
    scores = []

    for k in k_values:
        model = MiniBatchKMeans(n_clusters=k, random_state=10)
        labels = model.fit_predict(all_frames)
        score = calinski_harabasz_score(all_frames, labels)
        scores.append(score)

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, scores, marker='o')
    plt.title("Calinski-Harabasz Score vs Number of Clusters")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Calinski-Harabasz Score")
    plt.grid(True)
    plt.show()

#Find "elbow" point as the k with the max second derivative
    diff = np.gradient(scores, edge_order=2)
    second_diff = np.gradient(diff, edge_order=2)
    optimal_k_index = np.argmax(second_diff)
    optimal_k = k_values[optimal_k_index]

    print(f"Optimal number of clusters determined manually: {optimal_k}")'''

    '''# Elbow Method
    clusterer = MiniBatchKMeans(random_state=10)
    ##double check this, elbow plot yourself,
    temp = range(2, 40, 2)
    visualizer = KElbowVisualizer(clusterer, k=temp, metric = 'calinski_harabasz', locate_elbow=True, timings=False)
    visualizer.fit(all_frames)
    visualizer.show()
    optimal_k = visualizer.elbow_value_
    print(f"Optimal number of clusters determined by Elbow Method: {optimal_k}")'''
    '''from sklearn.metrics import silhouette_samples
    k_range = range(5, 10)

    for n_clusters in k_range:
        clusterer = MiniBatchKMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(all_frames)

    # Get individual silhouette scores
        sample_silhouette_values = silhouette_samples(all_frames, cluster_labels)

    # Bin the individual scores
        bins = np.arange(-1.0, 1.05, 0.1)  # range -1.0 to 1.0
        hist, bin_edges = np.histogram(sample_silhouette_values, bins=bins)

    # Plot the histogram for this k
        plt.figure(figsize=(8, 5))
        plt.bar(
            [f"{round(bins[i], 1)}–{round(bins[i+1], 1)}" for i in range(len(hist))],
            hist,
            width=0.8,
            edgecolor='black',
            align='center'
        )
        plt.xlabel("Silhouette Score Range")
        plt.ylabel("Number of Samples")
        plt.title(f"Silhouette Score Distribution for k = {n_clusters}")
        plt.xticks(rotation=45)
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()
        # Perform KMeans clustering
    print("Kmeans clustering ... (this might take a while)")'''
    import numpy as np
    import cv2
    from pathlib import Path
    from skimage import img_as_ubyte
    from sklearn.cluster import MiniBatchKMeans
    from umap import UMAP
    from sklearn.metrics import silhouette_score, davies_bouldin_score

    '''
    # dimensions to reduce testing
    save_root = Path(config).parents[0] / "cluster-test"
    save_root.mkdir(parents=True, exist_ok=True)
    umap_dims = [10, 100, 1000]
    k = 30
    video_path_list = list(video_paths.keys())
    clustered_frames = {}

    # Helper to map global index to frame number and video path
    def get_frame_info(global_index):
        for vid, (start, end) in enumerate(video_index_ranges):
            if start <= global_index < end:
                local_index = global_index - start
                frame_number = frame_indices_per_video[vid][local_index]
                video_path = video_path_list[vid]
                return int(frame_number), video_path
        return None, None
    

    for dim in umap_dims:
        print("Reducing dimensions with UMAP to", dim, "components...")
        umap = UMAP(n_components=dim, random_state=42)
        reduced = umap.fit_transform(all_frames)
        kmeans = MiniBatchKMeans(n_clusters = k, random_state=42)
        labels = kmeans.fit_predict(reduced)

        # Evaluation
        if len(set(labels)) > 1:
            sil_score = silhouette_score(reduced, labels)
            dbi_score = davies_bouldin_score(reduced, labels)
            print(f"UMAP Dim: {dim} | Silhouette Score: {sil_score:.4f} | Davies-Bouldin Index: {dbi_score:.4f}")
        else:
            print(f"UMAP Dim: {dim} | Only one cluster found, cannot compute Silhouette or DBI.")

        for cluster_id in range(k):
            print("Processing cluster", cluster_id, "in UMAP dimension", dim)
            cluster_indices = np.where(labels == cluster_id)[0]
            sampled_indices = np.random.choice(cluster_indices, size=min(3, len(cluster_indices)), replace=False)

            cluster_folder = save_root / f"umap_{dim}" / f"cluster_{cluster_id}"
            cluster_folder.mkdir(parents=True, exist_ok=True)

            for idx in sampled_indices:
                frame_number, video_path = get_frame_info(idx)
                print("Processing frame number", frame_number, "in video", video_path)
                if video_path is None:
                    continue

                cap = cv2.VideoCapture(video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                cap.release()

                if ret and frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = img_as_ubyte(frame)

                    video_name = Path(video_path).stem
                    filename = f"{video_name}_frame{frame_number}.png"
                    filepath = cluster_folder / filename
                    cv2.imwrite(str(filepath), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # List top-level directories where images were saved
    sorted(list(save_root.glob("*")))'''


    
    final_k = 30
    kmeans = MiniBatchKMeans(
        n_clusters=final_k, tol=1e-3, batch_size=batchsize, max_iter=max_iter
    )
    kmeans.fit(all_frames)
    # Reduce dimensions using PCA for visualization
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(all_frames)
        
        # Plot clustered frames
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=data_pca[:, 0], y=data_pca[:, 1], hue=kmeans.labels_, palette="Set1", s=10, alpha=0.8, legend=False)
        
        # Plot cluster centers
    cluster_centers = pca.transform(kmeans.cluster_centers_)
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c="black", s=100, edgecolors="white", marker="o")
        
        # Add train time and inertia text
    plt.text(min(data_pca[:, 0]), max(data_pca[:, 1]), f"inertia: {kmeans.inertia_:.6f}", fontsize=10)
        ##loop, find the optimal number of clustering, pick the best clustering
    plt.title("K-Means Frame Selection Visualization")
    plt.xticks([])
    plt.yticks([])
    plt.show()

        #added end


    ## working on this
    frames2pick = []
    print("Saving")
    for clusterid in range(final_k):
        clusterids = np.where(clusterid == kmeans.labels_)[0]
        if len(clusterids) > 0:
            frames2pick.append(np.random.choice(clusterids))

    if not frames2pick:
        print("Frame selection failed.")
        return

    print("Saving selected frames...")
    
    # Map each global index to a specific video + frame
    video_path_list = list(video_paths.keys())
    is_valid = []
    for idx in frames2pick:
        for vid, (start, end) in enumerate(video_index_ranges):
            if start <= idx < end:
                local_index = idx - start
                frame_number = frame_indices_per_video[vid][local_index]
                print(vid)
                print(video_path_list)
                video_path = video_path_list[vid]
                print('1')
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

        print("Kmeans clustering ... (this might take a while)")
        data = DATA - DATA.mean(axis=0)
        data = data.reshape(nframes, -1)  # stacking

        kmeans = MiniBatchKMeans(
            n_clusters=numframes2pick, tol=1e-3, batch_size=batchsize, max_iter=max_iter
        )
        kmeans.fit(data)
        ##pick 20 frames, find 20 clustering
        ##kmean still, what is a good number of clustering,
        ##5 good clustering, 4 images for each cluster, optimization of clustering number
        ##how diverse the data is, get a sense of how divserse, optimal number of clustering'''


import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.cluster import MiniBatchKMeans
import os

def update_cluster_space(
    new_frames,
    cluster_center_path="cluster_centers.npy",
    novelty_threshold=0.2,
    local_cluster_k=5,
    save_new_centers=True,
):
    """
    Update cluster centers with new frame features that are far from existing clusters.
    
    Parameters
    ----------
    new_frames : np.ndarray
        New frame features of shape (n_new_frames, n_features)

    cluster_center_path : str
        Path to existing .npy file containing cluster centers

    novelty_threshold : float
        Distance threshold above which a frame is considered novel

    local_cluster_k : int
        Number of clusters to use for local clustering of novel frames

    save_new_centers : bool
        If True, saves the updated cluster centers to disk

    Returns
    -------
    updated_centers : np.ndarray
        New full cluster center matrix

    novel_mask : np.ndarray
        Boolean array indicating which new frames were considered novel
    """
    assert new_frames.ndim == 2, "new_frames must be 2D"

    # Load or initialize cluster centers
    if os.path.exists(cluster_center_path):
        cluster_centers = np.load(cluster_center_path)
    else:
        cluster_centers = np.empty((0, new_frames.shape[1])) 

    # Compare new frames to existing centers
    if len(cluster_centers) > 0:
        _, distances = pairwise_distances_argmin_min(new_frames, cluster_centers)
        novel_mask = distances > novelty_threshold
    else:
        novel_mask = np.ones(len(new_frames), dtype=bool)

    novel_frames = new_frames[novel_mask]

    # Cluster novel frames 
    if len(novel_frames) > 0:
        k_local = min(local_cluster_k, len(novel_frames))
        local_kmeans = MiniBatchKMeans(n_clusters=k_local, random_state=42)
        local_kmeans.fit(novel_frames)
        new_centers = local_kmeans.cluster_centers_
    else:
        new_centers = np.empty((0, new_frames.shape[1]))

    # Update cluster centers
    updated_centers = np.concatenate([cluster_centers, new_centers], axis=0)

    if save_new_centers:
        np.save(cluster_center_path, updated_centers)

    return updated_centers, novel_mask



