import numpy as np
from collections import Counter

import matplotlib.pyplot as plt
from IPython.display import display

from graphviz import Digraph

import sklearn
from sklearn.model_selection import learning_curve
from sklearn import metrics

import torch
import torch.utils.data
import torchvision.transforms as transforms

from pycuda import autoinit, driver

#######################################################################################################################

def gpu_stat():
    if torch.cuda.is_available():

        def pretty_bytes(bytes, precision=1):
            abbrevs = ((1<<50, 'PB'),(1<<40, 'TB'),(1<<30, 'GB'),(1<<20, 'MB'),(1<<10, 'kB'),(1, 'bytes'))
            if bytes == 1:
                return '1 byte'
            for factor, suffix in abbrevs:
                if bytes >= factor:
                    break
            return '%.*f%s' % (precision, bytes / factor, suffix)

        device = autoinit.device
        print()
        print( 'GPU Name: %s' % device.name())
        print( 'GPU Memory: %s' % pretty_bytes(device.total_memory()))
        print( 'CUDA Version: %s' % str(driver.get_version()))
        print( 'GPU Free/Total Memory: %d%%' % ((driver.mem_get_info()[0] /driver.mem_get_info()[1]) * 100))

###################################################################################################################

def plot_learning_curves(m, loss_ylim=(0, 1.0), score_ylim=(0.0, 1.0), figsize=(14,6)):
    train_loss = m.values['train_loss']
    train_score = m.values['train_score']
    train_lr = m.values['train_lr']
    valid_loss = m.values['valid_loss']
    valid_score = m.values['valid_score']

    train_epochs = np.linspace(1, len(train_loss), len(train_loss))

    fig, ax = plt.subplots(1,2,figsize=figsize)

    if not train_loss is None:
        loss_train_min = np.min(train_loss)
        ax[0].plot(train_epochs, train_loss, color="r",
                   label="Trainings loss (min %.4f)" % loss_train_min) #alpha=0.3)

    if not valid_loss is None:
        loss_valid_min = np.min(valid_loss)
        ax[0].plot(train_epochs, valid_loss, color="b",
                   label="Validation loss (min %.4f)" % loss_valid_min) #alpha=0.3)
        ax[0].legend(loc="best")

    if not train_lr is None:
        ax0 = ax[0].twinx()
        ax0.plot(train_epochs, train_lr, color="g", label="Learning Rate") #alpha=0.3)
        ax0.set_ylabel('learning rate')

    ax[0].set_title("Loss")
    ax[0].set_xlim(0, np.max(train_epochs))
    ax[0].set_ylim(*loss_ylim)
    ax[0].set_xlabel('epochs')
    ax[0].set_ylabel('loss')

    if not train_score is None:
        score_train_max = np.max(train_score)
        ax[1].plot(train_epochs, train_score, color="r",
                   label="Trainings score (max %.4f)" % score_train_max)

    if not valid_score is None:
        score_valid_max = np.max(valid_score)
        ax[1].plot(train_epochs, valid_score, color="b",
                   label="Validation score (max %.4f)" % score_valid_max)

    if not train_lr is None:
        ax1 = ax[1].twinx()
        ax1.plot(train_epochs, train_lr, color="g", label="Learning Rate") #alpha=0.3)
        ax1.set_ylabel('learning rate')

    ax[1].set_title("Score")
    ax[1].set_xlim(0, np.max(train_epochs))
    ax[1].set_ylim(*score_ylim)
    ax[1].set_xlabel('epochs')
    ax[1].set_ylabel('score')
    ax[1].legend(loc="best")

    plt.grid(False)
    plt.tight_layout()

#####################################################################################################################

def plot_cross_validation_scores(scores, figsize=(12,4)):
    train_score = scores['train_score']
    valid_scores   = scores['test_score']
    score_difference = train_score - valid_scores

    plt.figure(figsize=figsize)
    plt.subplot(211)

    train_score_line, = plt.plot(train_score, color='r')
    valid_scores_line, = plt.plot(valid_scores, color='b')
    plt.ylabel("Score", fontsize="14")
    plt.legend([train_score_line, valid_scores_line], ["Train CV", "Validate CV"], bbox_to_anchor=(0, .4, .5, 0))
    plt.title("Train and Validation Cross Validation", x=.5, y=1.1, fontsize="15")

    # Plot bar chart of the difference.
    plt.subplot(212)
    difference_plot = plt.bar(range(len(score_difference)), score_difference)
    plt.xlabel("Cross-fold #")
    plt.legend([difference_plot], ["Test CV - Validation CV Score"], bbox_to_anchor=(0, 1, .8, 0))
    plt.ylabel("Score difference", fontsize="14")

    plt.show()

#####################################################################################################################

def plot_roc_curve(y_true, y_pred, y_proba):
    plt.figure()

    fpr, tpr, _ = metrics.roc_curve(y_true, y_proba[:, 1])
    plt.plot(fpr, tpr, color='red', label="predict_proba")

    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
    plt.plot(fpr, tpr, color='darkorange', label="predict")

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")

#####################################################################################################################
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
from matplotlib import cm
import itertools

def cm2inch(*tupl):
    '''
    Specify figure size in centimeter in matplotlib
    Source: http://stackoverflow.com/a/22787457/395857
    By gns-ank
    '''
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

def show_values(pc, fmt="%.2f", **kw):
    '''
    Heatmap with text in each cell with matplotlib's pyplot
    Source: http://stackoverflow.com/a/25074150/395857
    By HYRY
    '''
    pc.update_scalarmappable()
    ax = pc.axes
    for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)

def get_cmap():
    '''
    http://stackoverflow.com/questions/37517587/how-can-i-change-the-intensity-of-a-colormap-in-matplotlib
    '''
    cmap = cm.get_cmap('RdBu', 256) # set how many colors you want in color map
    # modify colormap
    alpha = 1.0
    colors = []
    for ind in range(cmap.N):
        c = []
        if ind<128 or ind> 210: continue
        for x in cmap(ind)[:3]: c.append(min(1,x*alpha))
        colors.append(tuple(c))
    my_cmap = matplotlib.colors.ListedColormap(colors, name = 'my_name')
    return my_cmap

def heatmap(AUC, title, xlabel, ylabel, xticklabels, yticklabels, ax, correct_orientation=False,
            cmap='RdBu', fmt="%.2f", graph_filepath='', normalize=False, remove_diagonal=False):
    '''
    Inspired by:
    - http://stackoverflow.com/a/16124677/395857
    - http://stackoverflow.com/a/25074150/395857
    '''
    if normalize:
        AUC = sklearn.preprocessing.normalize(AUC, norm='l1', axis=1)

    if remove_diagonal:
        matrix = np.copy(AUC)
        np.fill_diagonal(matrix, 0)
        if len(xticklabels)>2:
            matrix[:,-1] = 0
            matrix[-1, :] = 0
        values= matrix.flatten()
    else:
        values = AUC.flatten()
    vmin = values.min()
    vmax = values.max()

    #c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap='RdBu', vmin=0.0, vmax=1.0)
    c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap=get_cmap(), vmin=vmin, vmax=vmax)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

    # set tick labels
    ax.set_xticklabels(xticklabels, minor=False)
    ax.set_yticklabels(yticklabels, minor=False)

    # set title and x/y labels
    plt.title(title, y=1.08)

    plt.tight_layout()

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Remove last blank column
    plt.xlim( (0, AUC.shape[1]) )

    # Turn off all the ticks
    ax = plt.gca()
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # Add color bar
    plt.colorbar(c)

    # Add text in each cell
    show_values(c, fmt=fmt)

    # Proper orientation (origin at the top left instead of bottom left)
    if correct_orientation:
        ax.invert_yaxis()
        ax.xaxis.tick_top()

    if graph_filepath != '':
        plt.savefig(graph_filepath, dpi=300, format='png', bbox_inches='tight')
        plt.close()

def plot_classification_report(classification_report, title='Classification report ', cmap='RdBu',
                               figsize=(12,9), ax=None):
    '''
    Plot scikit-learn classification report.
    Extension based on http://stackoverflow.com/a/31689645/395857
    '''

    from matplotlib.cbook import MatplotlibDeprecationWarning
    import warnings
    warnings.simplefilter('ignore', MatplotlibDeprecationWarning)

    classes = []
    plotMat = []
    support = []
    class_names = []

    lines = classification_report.split('\n')
    for line in lines[2 : (len(lines) - 1)]:
        t = line.strip().replace('avg / total', 'micro-avg').split()
        if len(t) < 2: continue
        classes.append(t[0])
        v = [float(x)*100 for x in t[1: len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append(t[0])
        plotMat.append(v)

    xlabel = 'Metrics'
    ylabel = 'Classes'
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup) for idx, sup  in enumerate(support)]
    #    figure_width = 16
    #    figure_height = len(class_names) + 8
    correct_orientation = True

    # Plot it out
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = plt.gcf()
        fig.sca(ax)

    heatmap(np.array(plotMat), title, xlabel, ylabel, xticklabels, yticklabels, ax, correct_orientation, cmap=cmap)

    # resize
    #fig.set_size_inches(cm2inch(figsize[0], figsize[1]))

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens, ax=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if not ax is None:
        plt.gcf().sca(ax)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, y=1.08)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    ax.xaxis.set_label_position('top')

def plot_classifier_summary(y_true, y_pred, target_names, figsize=(12,5)):

    fig, ax = plt.subplots(1,2,figsize=figsize)

    plot_classification_report(classification_report(y_true, y_pred, target_names=target_names), ax=ax[0])
    plot_confusion_matrix(confusion_matrix(y_true, y_pred), target_names, False, ax=ax[1])

####################################################################################################################

from sklearn.manifold import TSNE
def plot_scatter_plots(X, y_pred, y_proba, y_true, target_names, figsize=(12,4)):

    tsne =TSNE(n_components=2, init='pca', random_state=0)
    tsne_data = tsne.fit_transform(X)

    idx = y_pred != y_true

    #set up figure
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=figsize)

    #Plot
    ax1.scatter(tsne_data[np.where(y_true==1),0],tsne_data[np.where(y_true==1),1],
                c='r', label=target_names[1])
    ax1.scatter(tsne_data[np.where(y_true==0),0],tsne_data[np.where(y_true==0),1],
                c='b', label=target_names[0])

    ax1.scatter(tsne_data[idx, 0], tsne_data[idx, 1], alpha=.8, lw=2, label="Error",
                facecolors='none', edgecolors='black', marker='o', s=80)

    ax2.scatter(y_proba[np.where(y_true==1),0],y_proba[np.where(y_true==1),1], c='r', label=target_names[1])
    ax2.scatter(y_proba[np.where(y_true==0),0],y_proba[np.where(y_true==0),1], c='b', label=target_names[0])

    ax2.scatter(y_proba[idx, 0], y_proba[idx, 1], alpha=.8, lw=2, label="Error",
                facecolors='none', edgecolors='black', marker='o', s=80)

    ax1.axes.get_xaxis().set_ticks([])
    ax1.axes.get_yaxis().set_ticks([])
    ax2.axes.get_xaxis().set_ticks([])
    ax2.axes.get_yaxis().set_ticks([])

    fig.suptitle('Scatter Plots', fontsize=20, fontweight='bold')
    plt.legend(loc=2, borderaxespad=.1, scatterpoints=1,bbox_to_anchor=(1.05, 1))

    fig.text(.25,.05,'TSNE Test Data', fontsize=15)
    fig.text(.65,.05,'CLF Proba Data', fontsize=15)

###################################################################################################################

def classifier_summary_report(X, y_true, y_pred, target_names):
    valid_score = metrics.f1_score(y_true, y_pred)
    acc_score = metrics.accuracy_score(y_true, y_pred)
    roc_score = metrics.roc_auc_score(y_true, y_pred)
    loss_score = metrics.log_loss(y_true, y_pred)

    print("Note: weighted average f1-score \n",
          metrics.classification_report(y_true, y_pred, target_names=target_names)
          )

    display(
        'Data points=%d' % X.shape[0],
        'Features=%d' % X.shape[1],
        'Class dist.=%f' % np.mean(y_true),
        'F1 valid=%f' % valid_score,
        'ACC=%f' % acc_score,
        'ROC_AUC=%f' % roc_score,
        'LOG_LOSS=%f' % loss_score,
        'Misclassified=%d' % np.sum(y_true != y_pred),
        'Data points=' + str([ i for (i, v) in enumerate(y_true != y_pred) if v][:20])
    )

###################################################################################################################

def class_info(classes):
    counts = Counter(classes)
    total = sum(counts.values())
    print("class percentages:")
    for cls in counts.keys():
        print("%6s: % 7d  =  % 5.1f%%" % (cls, counts[cls], counts[cls]/total*100))

def dataset_statistics(X_train, y_train, X_valid, y_valid, X_test, y_test, target_names):
    print("")
    print("Dataset statistics:")
    print("===================")
    print("%s %d" % ("number of features:".ljust(30), X_train.shape[1]))
    print("%s %d" % ("number of classes:".ljust(30), np.unique(y_train).shape[0]))
    print("%s %s" % ("data type:".ljust(30), X_train.dtype))
    print("%s %d (size=%dMB)"
          % ("number of train samples:".ljust(30), X_train.shape[0], int(X_train.nbytes / 1e6)))
    print("%s %d (size=%dMB)"
          % ("number of validation samples:".ljust(30), X_valid.shape[0], int(X_valid.nbytes / 1e6)))
    print("%s %d (size=%dMB)"
          % ("number of test samples:".ljust(30), X_test.shape[0], int(X_test.nbytes / 1e6)))
    print("%s %s" % ("classes".ljust(30) , str(target_names)))
    class_info(y_train)

###################################################################################################################

def plot_loss_curve(train_loss, train_score=None, valid_loss = None, valid_score=None, train_lr=None):
    train_epochs = np.linspace(1, len(train_loss), len(train_loss))

    fig, ax = plt.subplots(1,2,figsize=(14,6))

    if not train_loss is None:
        loss_train_min = np.min(train_loss)
        ax[0].plot(train_epochs, train_loss, color="r",
                   label="Trainings loss (min %.4f)" % loss_train_min) #alpha=0.3)

    if not valid_loss is None:
        loss_valid_min = np.min(valid_loss)
        ax[0].plot(train_epochs, valid_loss, color="b",
                   label="Validation loss (min %.4f)" % loss_valid_min) #alpha=0.3)

    if not train_lr is None:
        ax0 = ax[0].twinx()
        ax0.plot(train_epochs, train_lr, color="g", label="Learning Rate") #alpha=0.3)
        ax0.set_ylabel('lr')

    ax[0].set_title("Loss")
    ax[0].set_xlim(0, np.max(train_epochs))
    #     ax[0].set_ylim(0, 1)
    ax[0].set_xlabel('epochs')
    ax[0].set_ylabel('loss')
    ax[0].grid(True)
    ax[0].legend(loc="best")

    if not train_score is None:
        score_train_max = np.max(train_score)
        ax[1].plot(train_epochs, train_score, color="r",
                   label="Trainings score (max %.4f)" % score_train_max)

    if not valid_score is None:
        score_valid_max = np.max(valid_score)
        ax[1].plot(train_epochs, valid_score, color="b",
                   label="Validation score (max %.4f)" % score_valid_max)

    ax[1].set_title("Score")
    ax[1].set_xlim(0, np.max(train_epochs))
    ax[1].set_ylim(0.0, 1.02)
    ax[1].set_xlabel('epochs')
    ax[1].set_ylabel('score')
    ax[1].grid(True)
    ax[1].legend(loc="best")

    plt.legend(loc="best")

#####################################################################################################################

# https://stackoverflow.com/questions/42480111/model-summary-in-pytorch
# https://github.com/fchollet/keras/blob/master/keras/utils/layer_utils.py

def print_model_summary(model, line_length=None, positions=None, print_fn=print):
    """Prints a summary of a model.
    # Arguments
        model: model instance.
        line_length: Total length of printed lines
            (e.g. set this to adapt the display to different
            terminal window sizes).
        positions: Relative or absolute positions of log elements in each line.
            If not provided, defaults to `[.33, .55, .67, 1.]`.
        print_fn: Print function to use.
            It will be called on each line of the summary.
            You can set it to a custom function
            in order to capture the string summary.
    """

    line_length = line_length or 65
    positions = positions or [.45, .85, 1.]
    if positions[-1] <= 1:
        positions = [int(line_length * p) for p in positions]
    # header names for the different log elements
    to_display = ['Layer (type)', 'Shape', 'Param #']

    def print_row(fields, positions):
        line = ''
        for i in range(len(fields)):
            if i > 0:
                line = line[:-1] + ' '
            line += str(fields[i])
            line = line[:positions[i]]
            line += ' ' * (positions[i] - len(line))
        print_fn(line)

    print_fn( "Summary for model: " + model.__class__.__name__)
    print_fn('_' * line_length)
    print_row(to_display, positions)
    print_fn('=' * line_length)

    def print_module_summary(name, module):
        count_params = sum([np.prod(p.size()) for p in module.parameters()])
        output_shape = tuple([tuple(p.size()) for p in module.parameters()])
        cls_name = module.__class__.__name__
        fields = [name + ' (' + cls_name + ')', output_shape, count_params]
        print_row(fields, positions)

    module_count = len(set(model.modules()))
    for i, item in enumerate(model.named_modules()):
        name, module = item
        cls_name = str(module.__class__)
        if not 'torch' in cls_name or 'container' in cls_name:
            continue

        print_module_summary(name, module)
        if i == module_count - 1:
            print_fn('=' * line_length)
        else:
            print_fn('_' * line_length)

    trainable_count = 0
    non_trainable_count = 0
    for name, param in model.named_parameters():
        if 'bias' in name or 'weight' in name :
            trainable_count += np.prod(param.size())
        else:
            non_trainable_count += np.prod(param.size())

    print_fn('Total params:         {:,}'.format(trainable_count + non_trainable_count))
    print_fn('Trainable params:     {:,}'.format(trainable_count))
    print_fn('_' * line_length)

#####################################################################################################################

def layer_weight(data):
    mean = np.mean(data)
    std = np.std(data)

    hist, bins = np.histogram(data, bins=50)
    width = np.diff(bins)
    center = (bins[:-1] + bins[1:]) / 2

    return { 'mean':mean,
             'std':std,
             'hist':hist,
             'center':center,
             'width':width
             }

def plot_layer_stats(net):

    def to_np(x):
        return x.data.cpu().numpy()

    for name, module in net.named_modules():
        weight_attr = ['weight', 'weight_ih_l0', 'weight_hh_l0']
        weight_list = [w for w in weight_attr if hasattr(module, w)]

        bias_attr = ['bias', 'bias_ih_l0', 'bias_hh_l0']
        bias_list = [b for b in bias_attr if hasattr(module, b)]

        if not (weight_list and bias_list):
            continue

        for idx in range(len(weight_attr)):
            fig = plt.figure(idx, figsize=(10,4))

            if hasattr(module, weight_attr[idx]):
                if type(getattr(module, weight_attr[idx])) is torch.nn.parameter.Parameter:
                    w = layer_weight(to_np(getattr(module, weight_attr[idx])))

                    ax = plt.subplot2grid((1, 2), (0, 0))
                    ax.set_title("Module: %s-" % name + weight_attr[idx] +
                                 "\n Mean # %.4f" % w['mean'] + " STD # %.2e" % w['std'])
                    ax.bar(w['center'], w['hist'], align='center', width=w['width'])

            if hasattr(module, bias_attr[idx]):
                if type(getattr(module, bias_attr[idx])) is torch.nn.parameter.Parameter:
                    b = layer_weight(to_np(getattr(module, bias_attr[idx])))

                    ax = plt.subplot2grid((1, 2), (0, 1))
                    ax.set_title("Module: %s-" % name + bias_attr[idx] +
                                 "\n Mean # %.4f" % b['mean'] + " STD # %.2e" % b['std'])
                    ax.bar(b['center'], b['hist'], align='center', width=b['width'])

            plt.show();

#####################################################################################################################

def plot_model_graph(var, params):
    """ Produces Graphviz representation of PyTorch autograd graph

    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function

    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="8,8"))
    seen = set()

    def size_to_str(size):
        return '('+(', ').join(['%d'% v for v in size])+')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                node_name = '%s\n %s' % (param_map.get(id(u)), size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)
    add_nodes(var.grad_fn)
    return dot

###################################################################################################################

from imblearn.base import *
from imblearn.utils import check_ratio, check_target_type, hash_X_y
import logging


class OutlierSampler(SamplerMixin):
    def __init__(self, threshold=1.5, memory=None, verbose=0):
        self.threshold = threshold
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)

    def sample(self, X, y):
        # Check the consistency of X and y
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc'])

        check_is_fitted(self, 'X_hash_')
        self._check_X_y(X, y)

        X_out, y_out = self._sample(X, y)

        return X_out, y_out

    def _sample(self, X, y):
        outliers  = []
        for col in X.T: # loop over feature columns
            Q1 = np.percentile(col, 25)  # Calculate Q1 (25th percentile of the data) for the given feature
            Q3 = np.percentile(col, 75) # Calculate Q3 (75th percentile of the data) for the given feature

            step = self.threshold * (Q3 - Q1)  # Use the interquartile range to calculate an outlier step

            feature_outliers = np.where(~((col >= Q1 - step) & (col <= Q3 + step)))[0]
            outliers.extend(feature_outliers)

        # Find the data points that where considered outliers for more than one feature
        multi_feature_outliers = list((Counter(outliers) - Counter(set(outliers))).keys())

        X_out = np.delete(X, multi_feature_outliers, axis=0)
        y_out = np.delete(y, multi_feature_outliers, axis=0)

        if self.verbose:
            print('Sampled - reduced points form / to: ', X.shape, X_out.shape)
        return X_out, y_out

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc'])
        y = check_target_type(y)
        self.X_hash_, self.y_hash_ = hash_X_y(X, y)

        self._fit( X, y)

        return self

    def _fit(self, X, y):
        if self.verbose:
            print('OutlierSampler Fitted X/y: ', X.shape, y.shape)
        return self

    def fit_sample(self, X, y):
        return self.fit(X, y).sample(X, y)

###################################################################################################################


def visualize_data(img, label, figsize=None, ax=None):

    img = img.squeeze().cpu().numpy()

    if not figsize is None:
        plt.figure(figsize=figsize)

    if not ax:
        plt.title(label.item())
        plt.imshow(img, cmap='gray')
    else:
        ax.set_title(label.item())
        ax.imshow(img, cmap='gray')

#######################################################################################################################
