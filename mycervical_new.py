"""TODO(mycervical): Add a description here."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets.public_api as tfds

#Shared constants
_MYCERVICA_IMAGE_SIZE = 100
_MYCERVICA_IMAGE_SHAPE = (_MYCERVICA_IMAGE_SIZE, _MYCERVICA_IMAGE_SIZE, 3)

# Shared constants
#_CIFAR_IMAGE_SIZE = 100
#_CIFAR_IMAGE_SHAPE = (_CIFAR_IMAGE_SIZE, _CIFAR_IMAGE_SIZE, 3)

COLAB = False
XIAMEN = False
MACBOOK = True

# TODO(mycervical): BibTeX citation
_CITATION = """\
  @paul.xiong.2007@gmail.com
"""

# TODO(mycervical):
_DESCRIPTION = "The mycervical dataset consists of 10000 100x100 color"


class Mycervical(tfds.core.GeneratorBasedBuilder):
  """TODO(mycervical): Short description of my dataset."""

  # TODO(mycervical): Set up version.
  VERSION = tfds.core.Version('0.1.0')

  def _info(self):
    # TODO(mycervical): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        # This is the description that will appear on the datasets page.
        description=_DESCRIPTION,
        # tfds.features.FeatureConnectors
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            "image": tfds.features.Image(shape=_MYCERVICA_IMAGE_SHAPE),
            "label": tfds.features.ClassLabel(num_classes=10),
        }),
        # If there's a common (input, target) tuple from the features,
        # specify them here. They'll be used if as_supervised=True in
        # builder.as_dataset.
        supervised_keys=("image","label"),
        # Homepage of the dataset for documentation
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )
  @property
  def _mycervical_info(self):
    return MycervicalInfo(
        name=self.name,
        url="./data/cifar-10-binary.tar.gz",
        train_files=[
            "data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin",
            "data_batch_4.bin", "data_batch_5.bin"
        ],
        test_files=["test_batch.bin"],
        prefix="cifar-10-batches-bin/",
        label_files=["batches.meta.txt"],
        label_keys=["label"],
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    # TODO(mycervical): Downloads the data and defines the splits
    # dl_manager is a tfds.download.DownloadManager that can be used to
    # download and extract URLs
    '''return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            # These kwargs will be passed to _generate_examples
            gen_kwargs={},
        ),
    ]'''
    #boostx : commented following line to skip download and extract .gz.zip file. 
    # cifar_path = dl_manager.download_and_extract(self._cifar_info.url)
    
    if COLAB:
      mycervical_path = "/content/simclr-2/data/downloads/extracted/"
    elif XIAMEN:
      mycervical_path = "/notebooks/Paul/cervical/simclr-2/data/downloads/extracted/"
    elif MACBOOK:
      mycervical_path = "/Volumes/Bo500G32MCache/Cervical/simclr-2/data/downloads/extracted/"

      
    mycervical_info = self._mycervical_info

    mycervical_path = os.path.join(mycervical_path, mycervical_info.prefix)

    # Load the label names
    for label_key, label_file in zip(mycervical_info.label_keys,
                                     mycervical_info.label_files):
      labels_path = os.path.join(mycervical_path, label_file)
      with tf.io.gfile.GFile(labels_path) as label_f:
        label_names = [name for name in label_f.read().split("\n") if name]
      self.info.features[label_key].names = label_names

    # Define the splits
    def gen_filenames(filenames):
      for f in filenames:
        yield os.path.join(mycervical_path, f)

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            num_shards=10,  # Ignored when using a version with S3 experiment.
            gen_kwargs={"filepaths": gen_filenames(mycervical_info.train_files)}),
        #boostx : temp - commted follow TEST return
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            num_shards=1,  # Ignored when using a version with S3 experiment.
            gen_kwargs={"filepaths": gen_filenames(mycervical_info.test_files)}),
    ]

  def _generate_examples(self, filepaths):
    """Yields examples."""
    # TODO(mycervical): Yields (key, example) tuples from the dataset
    #yield 'key', {}
    """Generate CIFAR examples as dicts.

    Shared across CIFAR-{10, 100}. Uses self._cifar_info as
    configuration.

    Args:
      filepaths (list[str]): The files to use to generate the data.

    Yields:
      The cifar examples, as defined in the dataset info features.
    """
    label_keys = self._mycervical_info.label_keys
    index = 0  # Using index as key since data is always loaded in same order.
    for path in filepaths:
      for labels, np_image in _load_data(path, len(label_keys)):
        record = dict(zip(label_keys, labels))
        record["image"] = np_image
        yield index, record
        index += 1
        
class MycervicalInfo(collections.namedtuple("_MycervicalInfo", [
    "name",
    "url",
    "prefix",
    "train_files",
    "test_files",
    "label_files",
    "label_keys",
])):
  """Contains the information necessary to generate a CIFAR dataset.

  Attributes:
    name (str): name of dataset.
    url (str): data URL.
    prefix (str): path prefix within the downloaded and extracted file to look
      for `train_files` and `test_files`.
    train_files (list<str>): name of training files within `prefix`.
    test_files (list<str>): name of test files within `prefix`.
    label_files (list<str>): names of the label files in the data.
    label_keys (list<str>): names of the label keys in the data.
  """
  

def _load_data(path, labels_number=1):
  """Yields (labels, np_image) tuples."""
  with tf.io.gfile.GFile(path, "rb") as f:
    data = f.read()
  offset = 0
  max_offset = len(data) - 1
  while offset < max_offset:
    labels = np.frombuffer(data, dtype=np.uint8, count=labels_number,
                           offset=offset).reshape((labels_number,))
    # 1 byte per label, 1024 * 3 = 3072 bytes for the image.
    offset += labels_number
    img = (np.frombuffer(data, dtype=np.uint8, count=30000, offset=offset)
           .reshape((3, _MYCERVICA_IMAGE_SIZE, _MYCERVICA_IMAGE_SIZE))
           .transpose((1, 2, 0))
          )
    #offset += 3072
    offset += 30000
    yield labels, img
