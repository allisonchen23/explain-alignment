""" collection of various helper functions for running ACE"""

from multiprocessing import dummy as multiprocessing_dummy
import sys
import os
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import tcav.model as model
import torchvision
import torch
import numpy as np
from PIL import Image
import skimage.segmentation as segmentation
from skimage.segmentation import mark_boundaries
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
import tensorflow as tf
from functools import partial
from tqdm import tqdm

def load_features_model(arch,
                  n_classes,
                  device,
                  checkpoint_path=None):
    '''
    Build model from torchvision and load checkpoint. Return model and features model (cut off last layer)

    Arg(s):
        arch : str
            model architecture as specific in torchvision.models.__dict__
        n_classes : int
            number of classes to predict
        checkpoint_path : str or None
            path to restore model weights from
        device : torch.device
            Device to load model on

    Returns:
        model, features_model
            model : restored model
            features_model : model without the final classification layer
    '''
    model = torchvision.models.__dict__[arch](num_classes=n_classes)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        # Get rid of 'module' from the keys which is due to multi-GPU training
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
    model.eval()
    features_model = torch.nn.Sequential(*list(model.children())[:-1])
    features_model.eval()

    return model, features_model
    
def make_model(sess, model_to_run, model_path, 
               labels_path, randomize=False,):
  """Make an instance of a model.

  Args:
    sess: tf session instance.
    model_to_run: a string that describes which model to make.
    model_path: Path to models saved graph.
    randomize: Start with random weights
    labels_path: Path to models line separated class names text file.

  Returns:
    a model instance.

  Raises:
    ValueError: If model name is not valid.
  """
  if model_to_run == 'InceptionV3':
    mymodel = model.InceptionV3Wrapper_public(
        sess, model_saved_path=model_path, labels_path=labels_path)
  elif model_to_run == 'GoogleNet':
    # common_typos_disable
    mymodel = model.GoolgeNetWrapper_public(
        sess, model_saved_path=model_path, labels_path=labels_path)
  else:
    raise ValueError('Invalid model name')
  if randomize:  # randomize the network!
    sess.run(tf.global_variables_initializer())
  return mymodel


def load_image_from_file(filename, shape):
  """Given a filename, try to open the file. If failed, return None.
  Args:
    filename: location of the image file
    shape: the shape of the image file to be scaled
  Returns:
    the image if succeeds, None if fails.
  Rasies:
    exception if the image was not the right shape.
  """
  if not os.path.exists(filename):
    raise ValueError('Cannot find file: {}'.format(filename))
    # return None
  try:
    img = np.array(Image.open(filename).resize(
        shape, Image.BILINEAR))
    # Normalize pixel values to between 0 and 1.
    img = np.float32(img) / 255.0
    if not (len(img.shape) == 3 and img.shape[2] == 3):
      return None
    else:
      return img

  except Exception as e:
    tf.logging.info(e)
    return None
  return img


def load_images_from_files(filenames, max_imgs=500, return_filenames=False,
                           do_shuffle=True, run_parallel=True,
                           shape=(224, 224),
                           n_workers=100):
  """Return image arrays from filenames.
  Args:
    filenames: locations of image files.
    max_imgs: maximum number of images from filenames.
    return_filenames: return the succeeded filenames or not
    do_shuffle: before getting max_imgs files, shuffle the names or not
    run_parallel: get images in parallel or not
    shape: desired shape of the image
    n_workers: number of workers in parallelization.
  Returns:
    np.array : image arrays and succeeded filenames if return_filenames=True.
  """
  imgs = []
  # First shuffle a copy of the filenames.
  filenames = filenames[:]
  if do_shuffle:
    np.random.shuffle(filenames)
  if return_filenames:
    final_filenames = []
  if run_parallel:
    pool = multiprocessing_dummy.Pool(n_workers)
    # pool = multiprocessing.get_context("spawn").Pool(n_workers)
    # partial_fn = partial(load_image_from_file, shape=shape)
    # imgs = pool.map(partial_fn,
    #                 filenames[:max_imgs])
    imgs = pool.map(lambda filename: load_image_from_file(filename, shape),
                    filenames[:max_imgs])
    if return_filenames:
      final_filenames = [f for i, f in enumerate(filenames[:max_imgs])
                         if imgs[i] is not None]
    imgs = [img for img in imgs if img is not None]
  else:
    for filename in tqdm(filenames):
      img = load_image_from_file(filename, shape)
      if img is not None:
        imgs.append(img)
        if return_filenames:
          final_filenames.append(filename)
      if len(imgs) >= max_imgs:
        break

  if return_filenames:
    return np.array(imgs), final_filenames
  else:
    return np.array(imgs)


def get_acts_from_images(imgs, model, bottleneck_name):
  """Run images in the model to get the activations.
  Args:
    imgs: a list of images
    model: a model instance
    bottleneck_name: bottleneck name to get the activation from
  Returns:
    numpy array of activations.
  """
  return np.asarray(model.run_examples(imgs, bottleneck_name)).squeeze()

def extract_patch(image, 
                  mask, 
                  average_image_value, 
                  image_shape):
        """Extracts a patch out of an image.

        Args:
        image: The original image
        mask: The binary mask of the patch area
        average_image_value : float
        image_shape : tuple(int, int)
            size of image

        Returns:
        image_resized: The resized patch such that its boundaries touches the
        image boundaries
        patch: The original patch. Rest of the image is padded with average value
        """
        mask_expanded = np.expand_dims(mask, -1)
        patch = (mask_expanded * image + (
            1 - mask_expanded) * float(average_image_value) / 255)
        ones = np.where(mask == 1)
        h1, h2, w1, w2 = ones[0].min(), ones[0].max(), ones[1].min(), ones[1].max()
        image = Image.fromarray((patch[h1:h2, w1:w2] * 255).astype(np.uint8))
        image_resized = np.array(image.resize(image_shape,
                                              Image.BICUBIC)).astype(float) / 255
        return image_resized, patch

def return_superpixels(index_img, 
                       average_image_value,
                       image_shape,
                       method='slic',
                       param_dict=None):
        """Returns all patches for one image.

        Given an image, calculates superpixels for each of the parameter lists in
        param_dict and returns a set of unique superpixels by
        removing duplicates. If two patches have Jaccard similarity more than 0.5,
        they are concidered duplicates.

        Args:
            img: The input image
            method: superpixel method, one of slic, watershed, quichsift, or
            felzenszwalb
            param_dict: Contains parameters of the superpixel method used in the form
            of {'param1':[a,b,...], 'param2':[z,y,x,...], ...}. For instance
            {'n_segments':[15,50,80], 'compactness':[10,10,10]} for slic
            method.
            average_image_value : float
            image_shape : tuple(int, int)
                size of image
        Raises:
        ValueError: if the segementation method is invaled.

        Returns:
            (int, np.array, np.array)
                int : image index
                np.array : superpixel patches
                np.array : normal sized patches
        """
        # Passing in the index allows to use unordered maps
        idx, img = index_img

        if param_dict is None:
            param_dict = {}
        if method == 'slic':
            n_segmentss = param_dict.pop('n_segments', [15, 50, 80])
            n_params = len(n_segmentss)
            compactnesses = param_dict.pop('compactness', [20] * n_params)
            sigmas = param_dict.pop('sigma', [1.] * n_params)
        elif method == 'watershed':
            markerss = param_dict.pop('marker', [15, 50, 80])
            n_params = len(markerss)
            compactnesses = param_dict.pop('compactness', [0.] * n_params)
        elif method == 'quickshift':
            max_dists = param_dict.pop('max_dist', [20, 15, 10])
            n_params = len(max_dists)
            ratios = param_dict.pop('ratio', [1.0] * n_params)
            kernel_sizes = param_dict.pop('kernel_size', [10] * n_params)
        elif method == 'felzenszwalb':
            scales = param_dict.pop('scale', [1200, 500, 250])
            n_params = len(scales)
            sigmas = param_dict.pop('sigma', [0.8] * n_params)
            min_sizes = param_dict.pop('min_size', [20] * n_params)
        else:
            raise ValueError('Invalid superpixel method {}!')

        unique_masks = []
        for i in range(n_params):
            param_masks = []
            if method == 'slic':
                segments = segmentation.slic(
                    img, n_segments=n_segmentss[i], compactness=compactnesses[i],
                    sigma=sigmas[i])
            elif method == 'watershed':
                segments = segmentation.watershed(
                    img, markers=markerss[i], compactness=compactnesses[i])
            elif method == 'quickshift':
                segments = segmentation.quickshift(
                    img, kernel_size=kernel_sizes[i], max_dist=max_dists[i],
                    ratio=ratios[i])
            elif method == 'felzenszwalb':
                segments = segmentation.felzenszwalb(
                    img, scale=scales[i], sigma=sigmas[i], min_size=min_sizes[i])
            for s in range(segments.max()):
                mask = (segments == s).astype(float)
                if np.mean(mask) > 0.001:
                    unique = True
                    for seen_mask in unique_masks:
                        jaccard = np.sum(seen_mask * mask) / np.sum((seen_mask + mask) > 0)
                        if jaccard > 0.5:
                            unique = False
                            break
                    if unique:
                        param_masks.append(mask)
            unique_masks.extend(param_masks)

        superpixels, patches = [], []
        while unique_masks:
            superpixel, patch = extract_patch(
                img, 
                unique_masks.pop(),
                average_image_value=average_image_value,
                image_shape=image_shape)
            superpixels.append(superpixel)
            patches.append(patch)
        return idx, superpixels, patches

def get_features(dataset,
                 features_model,
                 device,
                 batch_size,
                 channel_mean):
        features = []

        features_model.eval()
        features_model = features_model.to(device)

        # Forward data through model in batches
        n_batches = int(dataset.shape[0] / batch_size) + 1
        with torch.no_grad():
            for batch_idx in tqdm(range(n_batches)):
                batch = dataset[batch_idx * batch_size : (batch_idx + 1) * batch_size]
                batch = torch.tensor(batch, dtype=torch.float)
                batch = torch.permute(batch, (0, 3, 1, 2))
                batch = batch.to(device)

                batch_features = features_model(batch).cpu().numpy()
                features.append(batch_features)
        features = np.concatenate(features, axis=0)

        # Flatten features to n_samples x feature_dim array either by taking mean across channels
        # Or expanding channel to 1D array
        if channel_mean and len(features.shape) > 3:
            features = np.mean(features, axis=(2, 3))
        else:
            features = np.reshape(features, [features.shape[0], -1])
        assert features.shape[0] == dataset.shape[0]

        return features

def flat_profile(cd, images, bottlenecks=None):
  """Returns concept profile of given images.

  Given a ConceptDiscovery class instance and a set of images, and desired
  bottleneck layers, calculates the profile of each image with all concepts and
  returns a profile vector

  Args:
    cd: The concept discovery class instance
    images: The images for which the concept profile is calculated
    bottlenecks: Bottleck layers where the profile is calculated. If None, cd
      bottlenecks will be used.

  Returns:
    The concepts profile of input images using discovered concepts in
    all bottleneck layers.

  Raises:
    ValueError: If bottlenecks is not in right format.
  """
  profiles = []
  if bottlenecks is None:
    bottlenecks = list(cd.dic.keys())
  if isinstance(bottlenecks, str):
    bottlenecks = [bottlenecks]
  elif not isinstance(bottlenecks, list) and not isinstance(bottlenecks, tuple):
    raise ValueError('Invalid bottlenecks parameter!')
  for bn in bottlenecks:
    profiles.append(cd.find_profile(str(bn), images).reshape((len(images), -1)))
  profile = np.concatenate(profiles, -1)
  return profile


def cross_val(a, b, methods):
  """Performs cross validation for a binary classification task.

  Args:
    a: First class data points as rows
    b: Second class data points as rows
    methods: The sklearn classification models to perform cross-validation on

  Returns:
    The best performing trained binary classification odel
  """
  x, y = binary_dataset(a, b)
  best_acc = 0.
  if isinstance(methods, str):
    methods = [methods]
  best_acc = 0.
  for method in methods:
    temp_acc = 0.
    params = [10**e for e in [-4, -3, -2, -1, 0, 1, 2, 3]]
    for param in params:
      clf = give_classifier(method, param)
      acc = cross_val_score(clf, x, y, cv=min(100, max(2, int(len(y) / 10))))
      if np.mean(acc) > temp_acc:
        temp_acc = np.mean(acc)
        best_param = param
    if temp_acc > best_acc:
      best_acc = temp_acc
      final_clf = give_classifier(method, best_param)
  final_clf.fit(x, y)
  return final_clf, best_acc


def give_classifier(method, param):
  """Returns an sklearn classification model.

  Args:
    method: Name of the sklearn classification model
    param: Hyperparameters of the sklearn model

  Returns:
    An untrained sklearn classification model

  Raises:
    ValueError: if the model name is invalid.
  """
  if method == 'logistic':
    return linear_model.LogisticRegression(C=param)
  elif method == 'sgd':
    return linear_model.SGDClassifier(alpha=param)
  else:
    raise ValueError('Invalid model!')


def binary_dataset(pos, neg, balanced=True):
  """Creates a binary dataset given instances of two classes.

  Args:
     pos: Data points of the first class as rows
     neg: Data points of the second class as rows
     balanced: If true, it creates a balanced binary dataset.

  Returns:
    The data points of the created data set as rows and the corresponding labels
  """
  if balanced:
    min_len = min(neg.shape[0], pos.shape[0])
    ridxs = np.random.permutation(np.arange(2 * min_len))
    x = np.concatenate([neg[:min_len], pos[:min_len]], 0)[ridxs]
    y = np.concatenate([np.zeros(min_len), np.ones(min_len)], 0)[ridxs]
  else:
    ridxs = np.random.permutation(np.arange(len(neg) + len(pos)))
    x = np.concatenate([neg, pos], 0)[ridxs]
    y = np.concatenate(
        [np.zeros(neg.shape[0]), np.ones(pos.shape[0])], 0)[ridxs]
  return x, y


def plot_concepts(cd, bn, num=10, address=None, mode='diverse', concepts=None):
  """Plots examples of discovered concepts.

  Args:
    cd: The concept discovery instance
    bn: Bottleneck layer name
    num: Number of images to print out of each concept
    address: If not None, saves the output to the address as a .PNG image
    mode: If 'diverse', it prints one example of each of the target class images
      is coming from. If 'radnom', randomly samples exmples of the concept. If
      'max', prints out the most activating examples of that concept.
    concepts: If None, prints out examples of all discovered concepts.
      Otherwise, it should be either a list of concepts to print out examples of
      or just one concept's name

  Raises:
    ValueError: If the mode is invalid.
  """
  if concepts is None:
    concepts = cd.dic[bn]['concepts']
  elif not isinstance(concepts, list) and not isinstance(concepts, tuple):
    concepts = [concepts]
  num_concepts = len(concepts)
  plt.rcParams['figure.figsize'] = num * 2.1, 4.3 * num_concepts
  fig = plt.figure(figsize=(num * 2, 4 * num_concepts))
  outer = gridspec.GridSpec(num_concepts, 1, wspace=0., hspace=0.3)
  for n, concept in enumerate(concepts):
    inner = gridspec.GridSpecFromSubplotSpec(
        2, num, subplot_spec=outer[n], wspace=0, hspace=0.1)
    concept_images = cd.dic[bn][concept]['images']
    concept_patches = cd.dic[bn][concept]['patches']
    concept_image_numbers = cd.dic[bn][concept]['image_numbers']
    if mode == 'max':
      idxs = np.arange(len(concept_images))
    elif mode == 'random':
      idxs = np.random.permutation(np.arange(len(concept_images)))
    elif mode == 'diverse':
      idxs = []
      while True:
        seen = set()
        for idx in range(len(concept_images)):
          if concept_image_numbers[idx] not in seen and idx not in idxs:
            seen.add(concept_image_numbers[idx])
            idxs.append(idx)
        if len(idxs) == len(concept_images):
          break
    else:
      raise ValueError('Invalid mode!')
    idxs = idxs[:num]
    for i, idx in enumerate(idxs):
      ax = plt.Subplot(fig, inner[i])
      ax.imshow(concept_images[idx])
      ax.set_xticks([])
      ax.set_yticks([])
      if i == int(num / 2):
        ax.set_title(concept)
      ax.grid(False)
      fig.add_subplot(ax)
      ax = plt.Subplot(fig, inner[i + num])
      mask = 1 - (np.mean(concept_patches[idx] == float(
          cd.average_image_value) / 255, -1) == 1)
      image = cd.discovery_images[concept_image_numbers[idx]]
      ax.imshow(mark_boundaries(image, mask, color=(1, 1, 0), mode='thick'))
      ax.set_xticks([])
      ax.set_yticks([])
      ax.set_title(str(concept_image_numbers[idx]))
      ax.grid(False)
      fig.add_subplot(ax)
  plt.suptitle(bn)
  if address is not None:
    with tf.gfile.Open(address + bn + '.png', 'w') as f:
      fig.savefig(f)
    plt.clf()
    plt.close(fig)


def cosine_similarity(a, b):
  """Cosine similarity of two vectors."""
  assert a.shape == b.shape, 'Two vectors must have the same dimensionality'
  a_norm, b_norm = np.linalg.norm(a), np.linalg.norm(b)
  if a_norm * b_norm == 0:
    return 0.
  cos_sim = np.sum(a * b) / (a_norm * b_norm)
  return cos_sim


def similarity(cd, num_random_exp=None, n_workers=25):
  """Returns cosine similarity of all discovered concepts.

  Args:
    cd: The ConceptDiscovery module for discovered conceps.
    num_random_exp: If None, calculates average similarity using all the class's
      random concepts. If a number, uses that many random counterparts.
    n_workers: If greater than 0, runs the function in parallel.

  Returns:
    A similarity dict in the form of {(concept1, concept2):[list of cosine
    similarities]}
  """

  def concepts_similarity(cd, concepts, rnd, bn):
    """Calcualtes the cosine similarity of concept cavs.

    This function calculates the pairwise cosine similarity of all concept cavs
    versus an specific random concept

    Args:
      cd: The ConceptDiscovery instance
      concepts: List of concepts to calculate similarity for
      rnd: a random counterpart
      bn: bottleneck layer the concepts belong to

    Returns:
      A dictionary of cosine similarities in the form of
      {(concept1, concept2): [list of cosine similarities], ...}
    """
    similarity_dic = {}
    for c1 in concepts:
      cav1 = cd.load_cav_direction(c1, rnd, bn)
      for c2 in concepts:
        if (c1, c2) in similarity_dic.keys():
          continue
        cav2 = cd.load_cav_direction(c2, rnd, bn)
        similarity_dic[(c1, c2)] = cosine_similarity(cav1, cav2)
        similarity_dic[(c2, c1)] = similarity_dic[(c1, c2)]
    return similarity_dic

  similarity_dic = {bn: {} for bn in cd.bottlenecks}
  if num_random_exp is None:
    num_random_exp = cd.num_random_exp
  randoms = ['random500_{}'.format(i) for i in np.arange(num_random_exp)]
  concepts = {}
  for bn in cd.bottlenecks:
    concepts[bn] = [cd.target_class, cd.random_concept] + cd.dic[bn]['concepts']
  for bn in cd.bottlenecks:
    concept_pairs = [(c1, c2) for c1 in concepts[bn] for c2 in concepts[bn]]
    similarity_dic[bn] = {pair: [] for pair in concept_pairs}
    def t_func(rnd):
      return concepts_similarity(cd, concepts[bn], rnd, bn)
    if n_workers:
      pool = multiprocessing.Pool(n_workers)
      sims = pool.map(lambda rnd: t_func(rnd), randoms)
    else:
      sims = [t_func(rnd) for rnd in randoms]
    while sims:
      sim = sims.pop()
      for pair in concept_pairs:
        similarity_dic[bn][pair].append(sim[pair])
  return similarity_dic


def save_ace_report(cd, accs, scores, address):
  """Saves TCAV scores.

  Saves the average CAV accuracies and average TCAV scores of the concepts
  discovered in ConceptDiscovery instance.

  Args:
    cd: The ConceptDiscovery instance.
    accs: The cav accuracy dictionary returned by cavs method of the
      ConceptDiscovery instance
    scores: The tcav score dictionary returned by tcavs method of the
      ConceptDiscovery instance
    address: The address to save the text file in.
  """
  report = '\n\n\t\t\t ---CAV accuracies---'
  for bn in cd.bottlenecks:
    report += '\n'
    for concept in cd.dic[bn]['concepts']:
      report += '\n' + bn + ':' + concept + ':' + str(
          np.mean(accs[bn][concept]))
  with tf.gfile.Open(address, 'w') as f:
    f.write(report)
  report = '\n\n\t\t\t ---TCAV scores---'
  for bn in cd.bottlenecks:
    report += '\n'
    for concept in cd.dic[bn]['concepts']:
      pvalue = cd.do_statistical_testings(
          scores[bn][concept], scores[bn][cd.random_concept])
      report += '\n{}:{}:{},{}'.format(bn, concept,
                                       np.mean(scores[bn][concept]), pvalue)
  with tf.gfile.Open(address, 'w') as f:
    f.write(report)


def save_concepts(cd, concepts_dir):
  """Saves discovered concept's images or patches.

  Args:
    cd: The ConceptDiscovery instance the concepts of which we want to save
    concepts_dir: The directory to save the concept images
  """
  for bn in cd.bottlenecks:
    for concept in cd.dic[bn]['concepts']:
      patches_dir = os.path.join(concepts_dir, bn + '_' + concept + '_patches')
      images_dir = os.path.join(concepts_dir, bn + '_' + concept)
      patches = (np.clip(cd.dic[bn][concept]['patches'], 0, 1) * 256).astype(
          np.uint8)
      images = (np.clip(cd.dic[bn][concept]['images'], 0, 1) * 256).astype(
          np.uint8)
      tf.gfile.MakeDirs(patches_dir)
      tf.gfile.MakeDirs(images_dir)
      image_numbers = cd.dic[bn][concept]['image_numbers']
      image_addresses, patch_addresses = [], []
      for i in range(len(images)):
        image_name = '0' * int(np.ceil(2 - np.log10(i + 1))) + '{}_{}'.format(
            i + 1, image_numbers[i])
        patch_addresses.append(os.path.join(patches_dir, image_name + '.png'))
        image_addresses.append(os.path.join(images_dir, image_name + '.png'))
      save_images(patch_addresses, patches)
      save_images(image_addresses, images)


def save_images(addresses, images):
  """Save images in the addresses.

  Args:
    addresses: The list of addresses to save the images as or the address of the
      directory to save all images in. (list or str)
    images: The list of all images in numpy uint8 format.
  """
  if not isinstance(addresses, list):
    image_addresses = []
    for i, image in enumerate(images):
      image_name = '0' * (3 - int(np.log10(i + 1))) + str(i + 1) + '.png'
      image_addresses.append(os.path.join(addresses, image_name))
    addresses = image_addresses
  assert len(addresses) == len(images), 'Invalid number of addresses'
  for address, image in zip(addresses, images):
    with tf.gfile.Open(address, 'w') as f:
      Image.fromarray(image).save(f, format='PNG')

