'''
Credit: https://github.com/rakhimovv/tcav/blob/master/cav.py
'''
import sys
import os.path
import pickle
import numpy as np
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
import tensorflow as tf
# import utils as utils

sys.path.insert(0, 'src')
from utils.utils import ensure_dir, informal_log


class CAV(object):
    """CAV class contains methods for concept activation vector (CAV).

    CAV represents semenatically meaningful vector directions in
    network's embeddings (bottlenecks).
    """

    @staticmethod
    def default_hparams():
        """HParams used to train the CAV.

        you can use logistic regression or linear regression, or different
        regularization of the CAV parameters.

        Returns:
          TF.HParams for training.
        """
        return {
            'model_type':'linear', 
            'alpha': 0.01
        }

    @staticmethod
    def load_cav(cav_path):
        """Make a CAV instance from a saved CAV (pickle file).

        Args:
          cav_path: the location of the saved CAV

        Returns:
          CAV instance.
        """
        with open(cav_path, 'rb') as pkl_file:
            save_dict = pickle.load(pkl_file)

        cav = CAV(save_dict['concepts'], save_dict['bottleneck'],
                  save_dict['hparams'], save_dict['saved_path'])
        cav.accuracies = save_dict['accuracies']
        cav.cavs = save_dict['cavs']
        return cav

    @staticmethod
    def cav_key(concepts, bottleneck, model_type, alpha):
        """A key of this cav (useful for saving files).

        Args:
          concepts: set of concepts used for CAV
          bottleneck: the bottleneck used for CAV
          model_type: the name of model for CAV
          alpha: a parameter used to learn CAV

        Returns:
          a string cav_key
        """
        return '-'.join([str(c) for c in concepts
                         ]) + '-' + bottleneck + '-' + model_type + '-' + str(alpha)

    @staticmethod
    def check_cav_exists(cav_dir, concepts, bottleneck, cav_hparams):
        """Check if a CAV is saved in cav_dir.

        Args:
          cav_dir: where cav pickles might be saved
          concepts: set of concepts used for CAV
          bottleneck: the bottleneck used for CAV
          cav_hparams: a parameter used to learn CAV

        Returns:
          True if exists, False otherwise.
        """
        cav_path = os.path.join(
            cav_dir,
            CAV.cav_key(concepts, bottleneck, cav_hparams['model_type'],
                        cav_hparams['alpha']) + '.pkl')
        return tf.gfile.Exists(cav_path)

    @staticmethod
    def _create_cav_training_set(concepts, bottleneck, acts):
        """Flattens acts, make mock-labels and returns the info.

        Labels are assigned in the order that concepts exists.

        Args:
            concepts: names of concepts
            bottleneck: the name of bottleneck where acts come from
            acts: a dictionary that contains activations
        Returns:
            x -  flattened acts
            labels - corresponding labels (integer)
            labels2text -  map between labels and text.
        """

        x = []
        labels = []
        labels2text = {}
        # to make sure postiive and negative examples are balanced,
        # truncate all examples to the size of the smallest concept.
        min_data_points = np.min(
            [acts[concept][bottleneck].shape[0] for concept in acts.keys()])

        for i, concept in enumerate(concepts):
            x.extend(acts[concept][bottleneck][:min_data_points].reshape(
                min_data_points, -1))
            labels.extend([i] * min_data_points)
            labels2text[i] = concept
        x = np.array(x)
        labels = np.array(labels)

        return x, labels, labels2text

    def __init__(self, 
                 concepts, 
                 bottleneck, 
                 hparams, 
                 save_path=None, 
                 log_path=None,
                 debug=False):
        """Initialize CAV class.

        Args:
			concepts: set of concepts used for CAV
			bottleneck: the bottleneck used for CAV
			hparams : dict 
				a parameter dictionary used to learn CAV
			save_path: where to save this CAV
			log_path: file to save logging to
        """
        self.concepts = concepts
        self.bottleneck = bottleneck
        self.hparams = hparams
        self.save_path = save_path
        self.log_path = log_path
        self.debug = debug

    def hparam_search(self, acts, Cs, save_linear_model=True):
        '''
        Train multiple CAVs based on C values to determine best C value

        Arg(s):
            acts : is a dictionary of activations. In particular, acts takes for of
                {'concept1':{'bottleneck name1':[...act array...],
                             'bottleneck name2':[...act array...],...
                 'concept2':{'bottleneck name1':[...act array...],
            Cs : list[float]
                list if regularization values
        '''
        assert self.hparams['model_type'] == 'logistic', "Hyperparameter search only supported for 'logistic' model type. Received '{}'".format(self.hparams['model_type'])
        
        x, labels, labels2text = CAV._create_cav_training_set(
            self.concepts, self.bottleneck, acts)
        
        best_overall_accuracy = -1
        best_accuracies = None
        # best_c = -1
        best_lm = None
        for C in Cs:
            # Instantiate linear model
            if self.hparams['model_type'] == 'linear':
                lm = linear_model.SGDClassifier(alpha=self.hparams.alpha, tol=1e-3, max_iter=1000)
            elif self.hparams['model_type'] == 'logistic':
                if 'model_params' in self.hparams:
                    lm = linear_model.LogisticRegression(C=C, **self.hparams['model_params'])
                else:
                    lm = linear_model.LogisticRegression(C=C)
            else:
                raise ValueError('Invalid hparams.model_type: {}'.format(
                    self.hparams.model_type))
            # Train linear model
            accuracies = self._train_lm(lm, x, labels, labels2text)
            # Compare overall accuracy to current best; update best if necessary
            if accuracies['overall'] > best_overall_accuracy:
                best_overall_accuracy = accuracies['overall']
                best_accuracies = accuracies
                # best_c = C
                best_lm = lm

        # After we find the best linear model, update attributes and save
        if len(best_lm.coef_) == 1:
            # if there were only two labels, the concept is assigned to label 0 by
            # default. So we flip the coef_ to reflect this.
            self.cavs = [-1 * best_lm.coef_[0], best_lm.coef_[0]]
        else:
            self.cavs = [c for c in best_lm.coef_]
        self.accuracies = best_accuracies
        self.linear_model = best_lm
        self._save_cavs(
            save_linear_model=save_linear_model
        )

            
    def train(self, acts, save_linear_model=True):
        """Train the CAVs from the activations.

        Args:
          acts: is a dictionary of activations. In particular, acts takes for of
                {'concept1':{'bottleneck name1':[...act array...],
                             'bottleneck name2':[...act array...],...
                 'concept2':{'bottleneck name1':[...act array...],
        Raises:
          ValueError: if the model_type in hparam is not compatible.
        """

        # tf.logging.info('training with alpha={}'.format(self.hparams.alpha))

        x, labels, labels2text = CAV._create_cav_training_set(
            self.concepts, self.bottleneck, acts)

        if self.hparams['model_type'] == 'linear':
            lm = linear_model.SGDClassifier(alpha=self.hparams.alpha, tol=1e-3, max_iter=1000)
        elif self.hparams['model_type'] == 'logistic':
            if 'model_params' in self.hparams:
                lm = linear_model.LogisticRegression(**self.hparams['model_params'])
            else:
                lm = linear_model.LogisticRegression()
        else:
            raise ValueError('Invalid hparams.model_type: {}'.format(
                self.hparams.model_type))
        # TODO: Include hparam search for regularization term (C)
        self.accuracies = self._train_lm(lm, x, labels, labels2text)
        if len(lm.coef_) == 1:
            # if there were only two labels, the concept is assigned to label 0 by
            # default. So we flip the coef_ to reflect this.
            self.cavs = [-1 * lm.coef_[0], lm.coef_[0]]
        else:
            self.cavs = [c for c in lm.coef_]
        self.linear_model = lm
        self._save_cavs(
            save_linear_model=save_linear_model
        )

    def get_direction(self, concept):
        """Get CAV direction.

        Args:
          concept: the conept of interest

        Returns:
          CAV vector.
        """
        return self.cavs[self.concepts.index(concept)]

    def _save_cavs(self,
                   save_linear_model=True):
        """Save a dictionary of this CAV to a pickle."""
        save_dict = {
            'concepts': self.concepts,
            'bottleneck': self.bottleneck,
            'hparams': self.hparams,
            'accuracies': self.accuracies,
            'cavs': self.cavs,
            'saved_path': self.save_path
        }
        if save_linear_model:
            save_dict.update({'linear_model': self.linear_model})
        if self.save_path is not None:
            # with tf.gfile.Open(self.save_path, 'w') as pkl_file:
            with open(self.save_path, 'wb') as pkl_file:
                pickle.dump(save_dict, pkl_file)
        else:
			# tf.logging.info('save_path is None. Not saving anything')
            informal_log("save_path is None. Not saving anything", self.log_path, timestamp=True)

    def _train_lm(self, lm, x, y, labels2text):
        """Train a model to get CAVs.

        Modifies lm by calling the lm.fit functions. The cav coefficients are then
        in lm._coefs.

        Args:
          lm: An sklearn linear_model object. Can be linear regression or
            logistic regression. Must support .fit and ._coef.
          x: An array of training data of shape [num_data, data_dim]
          y: An array of integer labels of shape [num_data]
          labels2text: Dictionary of text for each label.

        Returns:
          Dictionary of accuracies of the CAVs.

        """
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.33, stratify=y)
        # if you get setting an array element with a sequence, chances are that your
        # each of your activation had different shape - make sure they are all from
        # the same layer, and input image size was the same
        lm.fit(x_train, y_train)
        y_pred = lm.predict(x_test)
        # get acc for each class.
        num_classes = max(y) + 1
        acc = {}
        num_correct = 0
        for class_id in range(num_classes):
            # get indices of all test data that has this class.
            idx = (y_test == class_id)
            acc[labels2text[class_id]] = metrics.accuracy_score(
                y_pred[idx], y_test[idx])
            # overall correctness is weighted by the number of examples in this class.
            num_correct += (sum(idx) * acc[labels2text[class_id]])
        acc['overall'] = float(num_correct) / float(len(y_test))
        if self.debug:
            informal_log('acc per class %s' % (str(acc)), self.log_path, timestamp=True)
        # tf.logging.info('acc per class %s' % (str(acc)))
        return acc


def get_or_train_cav(concepts,
                     bottleneck,
                     acts,
                     cav_dir=None,
                     cav_hparams=None,
                     overwrite=False,
                     save_linear_model=True,
                     Cs_hparam_search=None,
                     log_path=None):
    """Gets, creating and training if necessary, the specified CAV.

    Assumes the activations already exists.

    Args:
      concepts: set of concepts used for CAV
              Note: if there are two concepts, provide the positive concept
                    first, then negative concept (e.g., ['striped', 'random500_1']
      bottleneck: the bottleneck used for CAV
      acts: dictionary contains activations of concepts in each bottlenecks
            e.g., acts[concept][bottleneck]
      cav_dir: a directory to store the results.
      cav_hparams: a parameter used to learn CAV
      overwrite: if set to True overwrite any saved CAV files.

    Returns:
      returns a CAV instance
    """

    if cav_hparams is None:
        cav_hparams = CAV.default_hparams()

    cav_path = None
    if cav_dir is not None:
        # utils.make_dir_if_not_exists(cav_dir)
        ensure_dir(cav_dir)
        cav_path = os.path.join(
            cav_dir,
            CAV.cav_key(concepts, bottleneck, cav_hparams['model_type'], cav_hparams['alpha'])
                .replace('/', '.') + '.pkl')

        if not overwrite and os.path.exists(cav_path):
            # tf.logging.info('CAV already exists: {}'.format(cav_path))
            informal_log('CAV already exists: {}'.format(cav_path), log_path, timestamp=True)
            cav_instance = CAV.load_cav(cav_path)
            return cav_instance

    # tf.logging.info('Training CAV {} - {} alpha {}'.format(
        # concepts, bottleneck, cav_hparams.alpha))
    
    cav_instance = CAV(concepts, bottleneck, cav_hparams, cav_path)
    if Cs_hparam_search is not None:
        informal_log('Training CAV {} - {} alpha {}\n Hyperparameter search over C: {}'.format(
            concepts, bottleneck, cav_hparams['alpha'], Cs_hparam_search), log_path, timestamp=True)
        cav_instance.hparam_search(
            acts={c: acts[c] for c in concepts},
            Cs=Cs_hparam_search,
            save_linear_model=save_linear_model
        )
    else:
        informal_log('Training CAV {} - {} alpha {}'.format(
            concepts, bottleneck, cav_hparams['alpha']), log_path, timestamp=True)
        cav_instance.train(
            acts={c: acts[c] for c in concepts},
            save_linear_model=save_linear_model)
    
    return cav_instance