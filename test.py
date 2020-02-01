# -*- coding:utf-8 -*- -
from sklearn.metrics import confusion_matrix  # 生成混淆矩阵函数
import sys
import random
import matplotlib.pyplot as plt  # 绘图库
import numpy as np
import matplotlib.cm as cm
import pandas as pd
from scipy.stats import pearsonr
import os
from IPython import embed


def plot_confusion_matrix(ma, labels_name, title):
    ma = ma.astype('float') / ma.sum(axis=1)[:, np.newaxis]  # 归一化
    plt.imshow(ma, interpolation='nearest', cmap=cm.Blues)  # 在特定的窗口上显示图像
    plt.title(title)  # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=90)  # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)  # 将标签印在y轴坐标上
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def compare_conf_mats_2(class_num, true_mat, weights, crowd_worker, wid, wn, wr, w_alpha):
    # normalize weights matrix between 0 and 1
    plt.figure(figsize=(22, 13))
    for i in range(len(true_mat)):
        w2_mat, w3_mat = None, None
        for j in range(3):
            ax = plt.subplot2grid((3, 6), (j, i))
            if j == 0:
                w1_mat = true_mat[i] / true_mat[i].sum(axis=1).reshape((class_num, 1))
                plt.imshow(w1_mat.transpose(1, 0), interpolation='nearest', cmap=cm.Blues, vmin=0, vmax=1)
                print(w1_mat)
                plt.title('Real', size=32)
                plt.xticks(fontsize=28)
                plt.yticks(fontsize=28)
                # center: 0.495, 0.5
                ax.text(0.002, 0.002, round(w1_mat[0][0], 2), va="center", ha="center", size=30, color='k')
                ax.text(0.988, 0.002, round(w1_mat[1][0], 2), va="center", ha="center", size=30, color='k')
                ax.text(0.002, 0.998, round(w1_mat[0][1], 2), va="center", ha="center", size=30, color='k')
                ax.text(0.988, 0.998, round(w1_mat[1][1], 2), va="center", ha="center", size=30, color='k')
            elif j == 1:
                w2_mat = weights[i].transpose(1, 0)
                plt.imshow(w2_mat, interpolation='nearest', cmap=cm.Blues, vmin=0, vmax=1)
                plt.title('SpeeLFC', size=32)
                plt.xticks(fontsize=28)
                plt.yticks(fontsize=28)
                this_title = 'ID:{}'.format(wid[i])
                plt.figtext(1 / 12 * (2 * i + 1), 0.95, this_title, va="center", ha="center", size=36)
                this_title = 'R:{}, L:{}'.format(wr[i], round(np.trace(w2_mat) / class_num, 2))
                ax.text(0.002, 0.002, round(w2_mat[0][0], 2), va="center", ha="center", size=30, color='k')
                ax.text(0.988, 0.002, round(w2_mat[1][0], 2), va="center", ha="center", size=30, color='k')
                ax.text(0.002, 0.998, round(w2_mat[0][1], 2), va="center", ha="center", size=30, color='k')
                ax.text(0.988, 0.998, round(w2_mat[1][1], 2), va="center", ha="center", size=30, color='k')
            else:
                w3_mat = crowd_worker[i]
                plt.imshow([[0, 0], [0, 0]], interpolation='nearest', cmap=cm.Blues, vmin=0, vmax=1)
                plt.title('Crowd-Layer', size=32)
                plt.xticks(fontsize=28)
                plt.yticks(fontsize=28)
                ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
                ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
                ax.tick_params(axis='x', which='minor', length=0)
                ax.tick_params(axis='x', which='minor', length=0)
                ax.grid(which='minor')
                ax.text(0.002, 0.002, round(w3_mat[0][0], 2), va="center", ha="center", size=30, color='k')
                ax.text(0.988, 0.002, round(w3_mat[0][1], 2), va="center", ha="center", size=30, color='k')
                ax.text(0.002, 0.998, round(w3_mat[1][0], 2), va="center", ha="center", size=30, color='k')
                ax.text(0.988, 0.998, round(w3_mat[1][1], 2), va="center", ha="center", size=30, color='k')
                ax.yaxis.grid(True, which='minor')
            plt.xticks(range(class_num), range(1, class_num + 1))
            plt.yticks(range(class_num), range(1, class_num + 1))
            plt.grid(True, which='minor')
    plt.subplots_adjust(left=0.02, bottom=0.02, top=0.9, right=0.99, wspace=0.3, hspace=0.1)

    plt.savefig('{}.pdf'.format('2_workers'), format='pdf')
    # plt.show()


def compare_conf_mats_10(class_num, true_mat, weights, learn_worker_shuang, wid, wn, wr, w_alpha):
    # normalize weights matrix between 0 and 1
    plt.figure(figsize=(22, 8))
    for i in range(len(true_mat)):
        w2_mat, w3_mat = None, None
        for j in range(2):
            plt.subplot2grid((2, 6), (j, i))
            if j == 0:
                w1_mat = true_mat[i] / true_mat[i].sum(axis=1).reshape((class_num, 1))
                plt.imshow(w1_mat.transpose(1, 0), interpolation='nearest', cmap=cm.Blues)
                plt.title('Real transition matrix', size=18)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
            elif j == 1:
                w2_mat = weights[i]
                plt.imshow(w2_mat, interpolation='nearest', cmap=cm.Blues)
                plt.title('Estimated by SpeeLFC', size=18)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                this_title = 'ID:{}'.format(wid[i])
                plt.figtext((1 - 0.01) / 12 * (2 * i + 1) + 0.005, 0.97, this_title, va="center", ha="center", size=22)
            else:
                # w3_mat = learn_worker_shuang[i]
                # plt.imshow(w3_mat, interpolation='nearest', cmap=cm.Blues)
                # plt.title('Learned with LFCDM-3', size=16)
                pass
            plt.xticks(range(class_num), range(1, class_num + 1))
            plt.yticks(range(class_num), range(1, class_num + 1))
    plt.subplots_adjust(left=0.02, bottom=0.03, top=0.91, right=0.98, wspace=0.2, hspace=0.15)

    plt.savefig('{}.pdf'.format('10_workers'), format='pdf')
    # plt.show()


def main():
    w_1 = np.load('D:\\aaai\\worker_hun_2.npy')
    w_2 = np.load('D:\\aaai\\2分类出图\\ds\\all_worker_l1.npy')[-1]
    # w_3 = np.load('C:\\Users\\isaac\\Desktop\\10分类出图\\shuang\\all_worker_l1.npy')[-1]
    w_acc = np.load('D:\\aaai\\worker_acc_2.npy')[()]
    workers = np.load('D:\\aaai\\workers_name.npz')['workers'].tolist()
    worker_alpha_dan = np.load('D:\\aaai\\2分类出图\\glad\\all_worker_alpha.npy')[-1]
    # worker_alpha_two = np.load('C:\\Users\\isaac\\Desktop\\10分类出图\\shuang\\all_worker_alpha.npy')[-1]
    print(worker_alpha_dan)
    worker_number = len(workers)
    x = []
    y = []
    for i in range(worker_number):
        wname = workers[i]
        x.append(w_acc[wname][0])
        y.append(np.trace(w_2[i]) / 10)
    # for index, i in enumerate(zip(x, y)):
    #     print(workers[index], i)
    print(pearsonr(x, y))

    # sys.exit()
    w_acc_sort = sorted(w_acc.items(), key=lambda xx: xx[1][1], reverse=True)
    print(w_acc_sort)

    # chose_id = sorted([0, 2, 37, 61, 65, 87])
    # chose_id = [5, 6, 11, 20, 37, 41]
    chose_id = []
    for index, i in w_acc_sort:
        chose_id.append(i[0])
        if index == 5:
            break
    for i in range(44):
        wname = workers[i]
        print(i, w_acc[wname], worker_alpha_dan[i])
    real_worker = []
    learn_worker = []
    learn_worker_shuang = []
    wids = []
    wn = []
    wr = []
    w_alpha = []
    for i in chose_id:
        wid = i
        wname = workers[wid]
        wids.append(wid)
        this_true = w_1[wid]
        zero_index = []
        for index_j, j in enumerate(this_true):
            if sum(j) == 0:
                zero_index.append(index_j)
        for j in zero_index:
            this_true[j] = [1, ] * 10
        real_worker.append(this_true)
        learn_worker.append(w_2[wid])
        learn_worker_shuang.append(0)
        wn.append(w_acc[wname][1])
        wr.append(round(w_acc[wname][0], 2))
        w_alpha.append(worker_alpha_dan[i])
        print(round(worker_alpha_dan[i], 2))
    compare_conf_mats_10(10, real_worker, learn_worker, learn_worker_shuang, wids, wn, wr, w_alpha)
    for i in chose_id:
        print(worker_alpha_dan[i])


def test():
    w_acc = np.load('C:\\Users\\isaac\\Desktop\\worker_acc.npy')[()]
    print(w_acc)
    times = []
    info = []
    for key, co in w_acc.items():
        times.append(co[1])
        info.append((co[1], co[0]))
    print(sorted(times))
    print(sorted(info, key=lambda x: x[0]))
    print(sorted(info, key=lambda x: x[1]))


def worker_dis():
    w_acc = np.load('D:\\aaai\\worker_acc_2.npy')[()]
    print(w_acc)
    acc = [x[0] for x in w_acc.values()]
    number = [x[1] for x in w_acc.values()]

    # df.boxplot()
    plt.figure(figsize=(11, 4))
    sp1 = plt.axes([0.09, 0.02, 0.15, 0.8])
    # plt.axes([0.1, 0.1, 0.15, 0.9])
    plt.boxplot(number,
                sym='+',
                widths=0.6,
                meanline=True, showmeans=True,
                meanprops={'color': 'b'},
                medianprops={'color': 'r'},
                boxprops={'color': 'k'},
                whiskerprops={'color': 'k', 'linestyle': '--'},
                capprops={'color': 'k'},
                flierprops={'color': 'k', 'markeredgecolor': 'k',
                            'markerfacecolor': 'none'})  # 也可用plot.box()
    plt.ylabel('num. answers', fontsize=18)
    plt.yticks(range(0, 5000, 1000), fontsize=17)
    plt.xticks([])
    plt.title('(a)', fontsize=20)
    plt.grid(b=False)

    sp2 = plt.axes([0.32, 0.02, 0.15, 0.8])
    plt.boxplot(acc,
                sym='+',
                widths=0.6,
                meanline=True, showmeans=True,
                meanprops={'color': 'b'},
                medianprops={'color': 'r'},
                boxprops={'color': 'k'},
                whiskerprops={'color': 'k', 'linestyle': '--'},
                capprops={'color': 'k'},
                flierprops={'color': 'k', 'markeredgecolor': 'k',
                            'markerfacecolor': 'none'})  # 也可用plot.box()
    plt.ylabel('accuracy', fontsize=18)
    plt.yticks(np.arange(0.2, 1.2, 0.2), fontsize=17)
    plt.xticks([])
    plt.title('(b)', fontsize=20)
    plt.grid(b=False)

    w_acc = np.load('D:\\aaai\\worker_acc.npy')[()]
    print(w_acc)
    acc = [x[0] for x in w_acc.values()]
    number = [x[1] for x in w_acc.values()]

    sp3 = plt.axes([0.61, 0.02, 0.15, 0.8])
    plt.boxplot(number,
                sym='+',
                widths=0.6,
                meanline=True, showmeans=True,
                meanprops={'color': 'b'},
                medianprops={'color': 'r'},
                boxprops={'color': 'k'},
                whiskerprops={'color': 'k', 'linestyle': '--'},
                capprops={'color': 'k'},
                flierprops={'color': 'k', 'markeredgecolor': 'k',
                            'markerfacecolor': 'none'})  # 也可用plot.box()
    plt.ylabel('num. answers', fontsize=18)
    plt.yticks(range(0, 400, 100), fontsize=17)
    plt.xticks([])
    plt.title('(a)', fontsize=20)
    plt.grid(b=False)

    sp4 = plt.axes([0.84, 0.02, 0.15, 0.8])
    plt.boxplot(acc,
                sym='+',
                widths=0.6,
                meanline=True, showmeans=True,
                meanprops={'color': 'b'},
                medianprops={'color': 'r'},
                boxprops={'color': 'k'},
                whiskerprops={'color': 'k', 'linestyle': '--'},
                capprops={'color': 'k'},
                flierprops={'color': 'k', 'markeredgecolor': 'k',
                            'markerfacecolor': 'none'})  # 也可用plot.box()
    plt.ylabel('accuracy', fontsize=18)
    plt.yticks(fontsize=17)
    plt.xticks([])
    plt.title('(b)', fontsize=20)
    plt.grid(b=False)

    plt.figtext(0.28, 0.96, "SPC dataset", va="center", ha="center", size=20)
    plt.figtext(0.8, 0.96, "MGC dataset", va="center", ha="center", size=20)

    plt.subplots_adjust(left=0.01, bottom=0.01, top=0.99, right=0.99, hspace=0.2)
    plt.savefig('{}.pdf'.format(1010), format='pdf')
    plt.show()


def acc_graph():
    train = np.load('C:\\Users\\isaac\\Desktop\\diff_10\\0\\all_train_acc.npy')
    test = np.load('C:\\Users\\isaac\\Desktop\\diff_10\\0\\all_test_acc.npy')
    # train2 = np.load('C:\\Users\\isaac\\Desktop\\maxmigtest\\128\\agg_acc.npy')
    # test2 = np.load('C:\\Users\\isaac\\Desktop\\maxmigtest\\128\\mw_acc.npy')
    # test3 = np.load('C:\\Users\\isaac\\Desktop\\maxmigtest\\128\\mv_acc.npy')
    x_axes = range(len(test))
    print(max(train))
    print(max(test))
    plt.plot(x_axes, train, linewidth=0.2)
    plt.plot(x_axes, test, c='r', linewidth=0.2)
    # plt.plot(x_axes, train2, c='k', linewidth=0.05)
    # plt.plot(x_axes, test2, c='g', linewidth=0.05)
    # plt.plot(x_axes, test3, c='y', linewidth=0.05)
    plt.xticks(range(0, 2200, 200))
    plt.yticks(np.arange(0, 1, 0.05))
    plt.grid(b=True)
    plt.savefig('{}.png'.format('acc'), format='png', dpi=1000)
    plt.show()


def worker_ds_graph():
    np.set_printoptions(precision=2)
    color = ['k', 'k', 'y', 'g', 'c', 'b', 'm', 'r', 'y', 'g']
    worker = np.load('C:\\Users\\isaac\\Desktop\\results_test\\121\\all_worker_l1.npy')
    w_acc = np.load('C:\\Users\\isaac\\Desktop\\worker_acc.npy')[()]
    workers = np.load('C:\\Users\\isaac\\Desktop\\music_data.npz')['workers'][()]
    w_1 = np.load('C:\\Users\\isaac\\Desktop\\worker_hun.npy')
    diagonal = np.diag_indices(10)
    worker_id = 5
    print(w_1[worker_id])
    print((w_1[worker_id] / w_1[worker_id].sum(axis=1)[:, np.newaxis])[diagonal])
    print(w_acc[workers[worker_id]])
    worker_1 = []
    for i in worker:
        worker_1.append(i[worker_id][diagonal])
    worker_1 = np.array(worker_1).transpose(1, 0)
    x_axes = range(worker_1.shape[1])
    for i in range(10):
        print(worker_1[i][1000], end=' ')
    print()
    for i in range(10):
        plt.plot(x_axes, worker_1[i], c=color[i], linewidth=0.5)
    plt.legend(range(10))
    plt.xticks(range(0, 1001, 100))
    plt.yticks(np.arange(0, 1, 0.1))
    plt.grid(b=True)
    plt.savefig('{}.png'.format('worker'), format='png', dpi=1000)
    plt.show()


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def beta():
    this_number = 9
    all_beta = np.load(os.path.join(r'C:\Users\isaac\Desktop\新参数\10', str(this_number), 'beta700.npy'))
    print(all_beta.shape)
    for index, i in enumerate(all_beta):
        if np.isnan(i).any():
            no_index = index - 1
            break
    else:
        no_index = all_beta.shape[0] - 1
    print(no_index)
    all_beta = all_beta[no_index].reshape(all_beta.shape[1])
    print(max(all_beta), min(all_beta))
    print(all_beta)
    plt.hist(all_beta, bins=np.arange(-1, 501, 5))
    plt.title("histogram")
    plt.savefig(os.path.join(r'C:\Users\isaac\Desktop\新参数分布图\10', str(this_number) + 'b' + str(no_index) + '.png'), format='png')
    plt.close()


def alpha():
    this_number = 6
    all_alpha = np.load(os.path.join(r'C:\Users\isaac\Desktop\新参数\10', str(this_number), 'all_worker_alpha.npy'))
    print(all_alpha.shape)
    for index, i in enumerate(all_alpha):
        if np.isnan(i).any():
            no_index = index - 1
            break
    else:
        no_index = all_alpha.shape[0] - 1
    print(no_index)
    all_alpha = np.tanh(all_alpha[no_index])
    print(max(all_alpha), min(all_alpha))
    plt.hist(all_alpha, bins=np.arange(-1, 1, 0.02))
    plt.savefig(os.path.join(r'C:\Users\isaac\Desktop\新参数分布图\10', str(this_number) + 'a' + str(no_index) + '.png'), format='png')
    plt.close()


def betaflow():
    all_beta = np.load(r'C:\Users\isaac\Desktop\新参数\10\6\beta700.npy')
    print(all_beta.shape)
    print(all_beta.max(), all_beta.min())
    nan_index = [np.isnan(x).any() for x in all_beta]
    if True in nan_index:
        no_index = nan_index.index(True) - 1
    else:
        no_index = -1
    print(no_index)
    all_beta = all_beta.reshape(1001, 700).transpose(1, 0)
    print(all_beta.shape)
    for i in range(50):
        this_index = random.randint(0, 700)
        print(this_index)
        plt.ylim(-1, 550)
        plt.plot(range(1001), all_beta[this_index])
        plt.savefig(os.path.join(r'C:\Users\isaac\Desktop\test', str(this_index) + '.png'), format='png')
        plt.close()


def alphaflow():
    all_alpha = np.load(r'C:\Users\isaac\Desktop\10分类2w轮\5\all_worker_alpha.npy')
    print(all_alpha.shape)
    print(np.nanmax(all_alpha), np.nanmin(all_alpha))
    nan_index = [np.isnan(x).any() for x in all_alpha]
    if True in nan_index:
        no_index = nan_index.index(True) - 1
    else:
        no_index = -1
    print(no_index)
    all_alpha = all_alpha.transpose(1, 0)
    print(all_alpha.shape)
    for i in range(44):
        this_index = i
        print(this_index)
        plt.ylim(-2, 11)
        plt.plot(range(20001), all_alpha[this_index])
        plt.savefig(os.path.join(r'C:\Users\isaac\Desktop\1w\10_2w_alphaflow\5', str(this_index) + '.jpg'), format='png')
        plt.close()


def tete():
    a = {'country.00005.mp3': 3.407106277536768, 'blues.00015.mp3': 0.4906219628730461,
         'jazz.00042.mp3': 0.4906219628730461, 'reggae.00056.mp3': 3.4103497596979655,
         'jazz.00071.mp3': 3.4103497596979655, 'classical.00024.mp3': 3.4103497596979655,
         'pop.00087.mp3': 3.410349759697965, 'pop.00047.mp3': 2.1879329128244844,
         'classical.00075.mp3': -1.862139195841008, 'reggae.00015.mp3': 3.4103497596979655,
         'pop.00082.mp3': 3.410349759697965, 'rock.00045.mp3': 2.1827403653515822,
         'reggae.00011.mp3': 1.0546459704023792, 'metal.00005.mp3': -0.8997635841157196,
         'blues.00023.mp3': 1.971451246518666, 'pop.00095.mp3': 5.447103202409577,
         'blues.00092.mp3': 3.4125287379551574, 'classical.00080.mp3': 1.8660381393429968,
         'classical.00085.mp3': 0.49062196287304616, 'jazz.00092.mp3': 2.2214813193157417,
         'blues.00004.mp3': 3.410349759697965, 'country.00084.mp3': -4.027825577860107,
         'metal.00047.mp3': -1.7279470150684435, 'reggae.00082.mp3': 1.0546459704023792,
         'reggae.00099.mp3': -0.012251226353106362, 'reggae.00041.mp3': -0.23167291423484096,
         'disco.00097.mp3': 3.408050002321556, 'classical.00011.mp3': 3.410502585106519,
         'pop.00066.mp3': 2.7430203021443873, 'classical.00081.mp3': 2.883619118184541,
         'jazz.00084.mp3': 3.412468610235584, 'rock.00030.mp3': 1.45789109045149, 'jazz.00058.mp3': 0.4906219628730461,
         'metal.00099.mp3': -1.3549387823648233, 'hiphop.00000.mp3': 1.0546459704023792,
         'metal.00043.mp3': 1.0546459704023792, 'hiphop.00023.mp3': 3.0846599933343186,
         'classical.00035.mp3': 1.801983589885297, 'pop.00027.mp3': 3.7367985181543815,
         'blues.00099.mp3': 1.0546459704023792, 'hiphop.00079.mp3': 3.412461805776269,
         'hiphop.00061.mp3': -0.8931121744909573, 'classical.00067.mp3': -0.7362253375338694,
         'country.00078.mp3': 1.0546459704023785, 'metal.00072.mp3': -0.06818929232704846,
         'country.00008.mp3': 5.447118267748718, 'rock.00039.mp3': -1.3663505925721113,
         'reggae.00081.mp3': -4.348945322467143, 'disco.00028.mp3': -1.428648073642827,
         'disco.00045.mp3': 3.7686646758559923, 'disco.00014.mp3': -2.0921751745839146,
         'classical.00034.mp3': -2.3364075691178994, 'rock.00011.mp3': 2.2507790797792846,
         'reggae.00014.mp3': 2.9411706297389975, 'country.00079.mp3': 1.0546459704023785,
         'country.00072.mp3': 1.9966796722587026, 'country.00091.mp3': 3.410349759697965,
         'country.00076.mp3': 2.746708030662504, 'pop.00090.mp3': 1.0546459704023792,
         'blues.00097.mp3': 3.410349759697965, 'reggae.00062.mp3': 1.0546459704023792,
         'classical.00064.mp3': 5.406803587741076, 'pop.00083.mp3': 3.7685974182183153,
         'country.00029.mp3': 3.103354873362079, 'classical.00060.mp3': 0.4904542929541864,
         'hiphop.00072.mp3': -2.485350763070447, 'rock.00027.mp3': -1.8383161247955688,
         'country.00092.mp3': 3.7338270016665467, 'metal.00009.mp3': 5.4471614860253785,
         'jazz.00070.mp3': 0.38762883406070436, 'country.00077.mp3': -0.6879951771318107,
         'pop.00096.mp3': 3.408050002321556, 'disco.00021.mp3': 0.9212090566925425,
         'classical.00027.mp3': 3.103853780318865, 'metal.00092.mp3': -2.344305238843238,
         'reggae.00094.mp3': -5.406046024055523, 'rock.00051.mp3': -4.393150621434163,
         'disco.00088.mp3': 6.001011428528136, 'classical.00069.mp3': 2.688997144173538,
         'rock.00071.mp3': 1.204224118457636, 'rock.00018.mp3': 2.7430203021443873,
         'disco.00005.mp3': 4.2577362001873595, 'reggae.00035.mp3': -1.0434780045812233,
         'jazz.00080.mp3': 3.4070860774325578, 'rock.00093.mp3': 0.49062196287304494,
         'classical.00059.mp3': 3.4071036718127017, 'jazz.00044.mp3': 3.4518829183359974,
         'disco.00098.mp3': -0.012251226353106283, 'rock.00009.mp3': -1.074920695059385,
         'hiphop.00053.mp3': -5.3329185111600435, 'country.00025.mp3': -3.3753920964644415,
         'disco.00004.mp3': 0.49062196287304494, 'pop.00009.mp3': -4.027825577860107,
         'disco.00001.mp3': 3.410349759697965, 'metal.00098.mp3': 0.4906219628730461,
         'pop.00003.mp3': 3.408050002321556, 'country.00093.mp3': 0.490621962873045,
         'pop.00034.mp3': 1.0546459704023792, 'disco.00050.mp3': 0.27385900905977273,
         'rock.00005.mp3': 3.7707120238607508, 'hiphop.00022.mp3': 3.412029764940096,
         'classical.00030.mp3': 1.0546459704023792, 'metal.00022.mp3': -4.22983782957111,
         'reggae.00018.mp3': 3.407135655362566, 'jazz.00035.mp3': -0.5140002304541391,
         'reggae.00026.mp3': 1.3777380804915782, 'country.00018.mp3': -1.7357587132950252,
         'hiphop.00014.mp3': 2.881670821795245, 'classical.00008.mp3': -2.0921751745839146,
         'hiphop.00093.mp3': 0.9425522799562548, 'disco.00020.mp3': -4.7006038596627855,
         'classical.00071.mp3': 1.0546459704023792, 'jazz.00094.mp3': 3.4121344588216833,
         'classical.00007.mp3': 0.49062196287304616, 'rock.00060.mp3': -3.4243676063854847,
         'disco.00058.mp3': 1.3613820186670753, 'reggae.00046.mp3': 3.4501900980097164,
         'rock.00062.mp3': -3.67112392526432, 'pop.00077.mp3': 3.410349759697965, 'jazz.00064.mp3': -1.2178885767691174,
         'blues.00069.mp3': 3.4103497596979655, 'rock.00033.mp3': 0.8408703385268644,
         'jazz.00068.mp3': 3.4103497596979655, 'blues.00089.mp3': 3.770677818577684,
         'jazz.00056.mp3': 3.4518701838241648, 'reggae.00063.mp3': 2.733948422648014,
         'rock.00063.mp3': 3.408050002321556, 'jazz.00051.mp3': 2.6849958588749483,
         'reggae.00086.mp3': -4.027825577860107, 'pop.00029.mp3': 1.0546459704023792,
         'hiphop.00025.mp3': 3.407134710160069, 'jazz.00021.mp3': 3.4103497596979655,
         'classical.00078.mp3': 3.4103497596979655, 'hiphop.00077.mp3': 3.0846599933343186,
         'blues.00002.mp3': -2.1006536636631794, 'classical.00050.mp3': 2.3331428276600286,
         'blues.00088.mp3': 3.410349759697965, 'pop.00060.mp3': 3.410349759697965,
         'reggae.00074.mp3': 3.1033686433527374, 'blues.00048.mp3': -0.08718565074778481,
         'classical.00049.mp3': 0.49062196287304616, 'disco.00000.mp3': 3.0846599933343186,
         'rock.00052.mp3': -5.078741513212588, 'jazz.00048.mp3': 3.408050002321556, 'pop.00002.mp3': 5.406902713593346,
         'rock.00068.mp3': -2.395781475731421, 'hiphop.00089.mp3': 1.0546459704023792,
         'metal.00071.mp3': 3.4103497596979655, 'metal.00052.mp3': -2.115495168199779,
         'rock.00056.mp3': 2.200797927410846, 'country.00074.mp3': -3.692867436325796,
         'disco.00037.mp3': 2.940797528460327, 'classical.00033.mp3': 2.733948422648014,
         'disco.00041.mp3': 3.407096658017073, 'disco.00056.mp3': -1.055845260652848,
         'jazz.00014.mp3': 2.192807993523653, 'country.00026.mp3': 1.0546459704023785,
         'disco.00038.mp3': -2.971499570315228, 'hiphop.00016.mp3': 3.407106277536768,
         'reggae.00088.mp3': 0.4899576012164494, 'reggae.00067.mp3': 1.0546459704023792,
         'classical.00047.mp3': 3.4118395566911612, 'metal.00083.mp3': 0.38762883406070425,
         'reggae.00077.mp3': 1.0546459704023792, 'reggae.00043.mp3': 0.38762883406070425,
         'jazz.00006.mp3': 2.399618217034956, 'country.00015.mp3': 3.410349759697965,
         'pop.00028.mp3': 3.4117956062087487, 'metal.00023.mp3': 1.0546459704023792,
         'hiphop.00001.mp3': 3.4121344588216833, 'hiphop.00076.mp3': 1.0546459704023792,
         'rock.00094.mp3': 3.410349759697965, 'disco.00092.mp3': 2.733948422648014,
         'jazz.00019.mp3': 0.9901617750425615, 'hiphop.00041.mp3': 2.700569867600356,
         'pop.00037.mp3': -0.13928472516421478, 'classical.00095.mp3': 2.7316225644688066,
         'disco.00060.mp3': 0.8357832901737087, 'reggae.00007.mp3': 5.406803587741076,
         'classical.00020.mp3': 5.406817986323013, 'jazz.00066.mp3': -0.49232778445594305,
         'jazz.00011.mp3': 3.412029764940096, 'country.00033.mp3': 3.408050002321556,
         'rock.00058.mp3': -2.9139948132755675, 'rock.00087.mp3': 0.4906219628730461,
         'pop.00004.mp3': 2.9390946636491586, 'blues.00067.mp3': 0.13525967735387862,
         'classical.00055.mp3': 1.0531615255844748, 'rock.00092.mp3': 0.2320608106853457,
         'blues.00085.mp3': 5.406807671445527, 'metal.00096.mp3': 0.4906219628730464,
         'metal.00086.mp3': 3.4103497596979655, 'metal.00028.mp3': 0.4906219628730461,
         'jazz.00079.mp3': -0.8362104001691418, 'classical.00003.mp3': 3.408050002321556,
         'blues.00070.mp3': 0.4906219628730461, 'rock.00069.mp3': 0.49062196287304494,
         'hiphop.00056.mp3': 3.4121830166444314, 'pop.00032.mp3': 3.408050002321556,
         'disco.00077.mp3': 2.940797528460327, 'blues.00081.mp3': 0.4906219628730461,
         'blues.00022.mp3': 2.6808635710106556, 'hiphop.00020.mp3': 3.4103497596979655,
         'pop.00042.mp3': 0.38762883406070436, 'disco.00003.mp3': 1.722287309533834,
         'hiphop.00013.mp3': 1.0546459704023792, 'jazz.00031.mp3': 2.802291403401095,
         'metal.00013.mp3': 2.737985573355352, 'pop.00012.mp3': 3.410349759697965,
         'country.00056.mp3': -3.202873931139149, 'reggae.00090.mp3': -0.989247631753859,
         'classical.00023.mp3': -1.021484656704097, 'reggae.00022.mp3': 2.7394172723996055,
         'jazz.00026.mp3': 3.408050002321556, 'blues.00051.mp3': -1.5738001227070548,
         'metal.00088.mp3': -0.48232464379217344, 'jazz.00093.mp3': 5.515148185355383,
         'country.00049.mp3': 3.4117917634818506, 'country.00064.mp3': -1.4421082848872955,
         'disco.00065.mp3': 1.7863556302763928, 'classical.00058.mp3': 0.49062196287304616,
         'country.00075.mp3': 3.408050002321556, 'hiphop.00069.mp3': 2.20985891909015,
         'jazz.00027.mp3': 0.9901618753936393, 'rock.00007.mp3': 2.318009880152254,
         'blues.00060.mp3': 0.23206081068534562, 'classical.00062.mp3': 0.49062196287304616,
         'blues.00003.mp3': 4.2588754099809485, 'metal.00007.mp3': -0.4077641937702615,
         'rock.00088.mp3': 3.408050002321556, 'rock.00073.mp3': 0.49062196287304494,
         'country.00088.mp3': 1.834003615691923, 'metal.00010.mp3': -4.027825577860107,
         'country.00041.mp3': 3.0846599933343186, 'rock.00020.mp3': 0.49062196287304494,
         'reggae.00002.mp3': 4.257765741158203, 'rock.00024.mp3': 3.4103497596979655,
         'disco.00081.mp3': 2.73722723682096, 'country.00057.mp3': -2.283038037101485,
         'metal.00042.mp3': 3.408050002321556, 'country.00000.mp3': 0.5856591870601711,
         'pop.00048.mp3': 3.410349759697965, 'jazz.00012.mp3': 1.3110207229816102, 'pop.00092.mp3': 3.410349759697965,
         'rock.00021.mp3': 1.9966796722587026, 'blues.00077.mp3': 1.0546459704023792,
         'disco.00078.mp3': 0.49062196287304494, 'metal.00073.mp3': 0.4906219628730461,
         'reggae.00047.mp3': -3.3753920964644415, 'jazz.00087.mp3': 3.4125287379551574,
         'reggae.00029.mp3': 3.4213496509461505, 'blues.00071.mp3': -0.4154342232787303,
         'metal.00031.mp3': -1.825372801236366, 'blues.00018.mp3': 2.3107381584238387,
         'pop.00030.mp3': 3.411793260574253, 'rock.00036.mp3': -4.4052388582765065,
         'disco.00071.mp3': -0.8550136182419689, 'blues.00036.mp3': -3.816652910738003,
         'hiphop.00057.mp3': 3.4103497596979655, 'metal.00077.mp3': 1.0546459704023792,
         'disco.00052.mp3': 3.410349759697965, 'pop.00097.mp3': -0.4134689755656835,
         'hiphop.00067.mp3': 2.733948422648014, 'hiphop.00063.mp3': 0.22990715815147839,
         'disco.00087.mp3': -1.1833639615267373, 'disco.00068.mp3': 2.162550991932866,
         'reggae.00095.mp3': -5.457552640058183, 'classical.00099.mp3': 3.4103497596979655,
         'metal.00093.mp3': 0.6351512134792718, 'blues.00016.mp3': 3.1037337488053973,
         'hiphop.00068.mp3': 2.9411706297389975, 'blues.00050.mp3': 3.411793260574253,
         'jazz.00023.mp3': 0.4904542929541864, 'pop.00078.mp3': -2.2012931453265305,
         'classical.00038.mp3': 1.0546459704023792, 'rock.00025.mp3': 1.0546459704023792,
         'classical.00057.mp3': 4.094378238960014, 'hiphop.00046.mp3': 4.091262751369023,
         'reggae.00072.mp3': 3.4121830166444314, 'pop.00070.mp3': -0.5471345741625517,
         'jazz.00072.mp3': 3.408050002321556, 'disco.00089.mp3': 3.4123144356586144,
         'jazz.00062.mp3': 0.9901618753936393, 'jazz.00036.mp3': 3.408050002321556, 'pop.00084.mp3': 2.6813429023518154,
         'blues.00076.mp3': -5.4169733872781025, 'classical.00063.mp3': 5.447139797534189,
         'rock.00065.mp3': 1.0546459704023792, 'classical.00043.mp3': 0.4704705381478776,
         'reggae.00064.mp3': 3.407188225272765, 'classical.00036.mp3': 4.09439495382081,
         'jazz.00029.mp3': 1.0546459704023792, 'classical.00086.mp3': 3.0845578025965685,
         'country.00022.mp3': 2.7430203021443873, 'reggae.00019.mp3': 3.4103497596979655,
         'rock.00000.mp3': 0.4906219628730461, 'country.00051.mp3': 3.410349759697965,
         'reggae.00098.mp3': -3.3272146161968967, 'country.00061.mp3': 2.736168711866367,
         'jazz.00047.mp3': 4.257689232780961, 'country.00016.mp3': -3.4107125614471117,
         'classical.00094.mp3': -3.3753920964644415, 'country.00098.mp3': 2.736168711866367,
         'rock.00001.mp3': -5.069944469255867, 'pop.00021.mp3': 3.408050002321556, 'blues.00057.mp3': 2.883619118184541,
         'reggae.00033.mp3': 1.8354525536105337, 'rock.00004.mp3': 3.408050002321556,
         'metal.00000.mp3': 3.412086847042682, 'country.00039.mp3': -5.560258006411917,
         'disco.00067.mp3': 3.408050002321556, 'metal.00076.mp3': 3.4103497596979655,
         'hiphop.00086.mp3': 2.9411706297389975, 'reggae.00039.mp3': 0.005030169819657169,
         'hiphop.00030.mp3': 1.0546459704023792, 'classical.00026.mp3': 2.311393841901294,
         'rock.00086.mp3': 0.4906219628730461, 'metal.00030.mp3': 3.408050002321556,
         'disco.00086.mp3': 0.490621962873045, 'pop.00006.mp3': -2.4444514806251867,
         'hiphop.00062.mp3': -0.012251226353106276, 'disco.00064.mp3': 0.2315980247206321,
         'jazz.00099.mp3': 3.4117938335195266, 'rock.00008.mp3': 3.3724714044062325,
         'disco.00053.mp3': -1.7031560882931902, 'reggae.00038.mp3': 3.408050002321556,
         'pop.00094.mp3': 3.411793256998652, 'country.00096.mp3': 0.3876288340607044,
         'country.00007.mp3': 3.4125287379551574, 'jazz.00057.mp3': -1.9339870503341459,
         'classical.00093.mp3': 2.741121157843095, 'classical.00048.mp3': 1.9564983430273624,
         'reggae.00085.mp3': 0.4906219628730461, 'jazz.00098.mp3': 2.6849958588749483,
         'pop.00074.mp3': -3.3753920964644415, 'hiphop.00091.mp3': 3.407173538947096,
         'classical.00001.mp3': 2.7391202284208034, 'hiphop.00071.mp3': 3.408050002321556,
         'disco.00073.mp3': 1.0546459704023792, 'hiphop.00052.mp3': 3.4103497596979655,
         'country.00062.mp3': 3.4121344588216833, 'metal.00053.mp3': 3.4121344588216833,
         'jazz.00053.mp3': 0.8148403565688415, 'country.00027.mp3': -3.4291738584310534,
         'jazz.00089.mp3': 1.4751502558665033, 'blues.00059.mp3': 3.4214164939903804,
         'classical.00014.mp3': 2.3331428276600286, 'pop.00085.mp3': 3.408050002321556,
         'metal.00037.mp3': -0.01990335090350465, 'reggae.00032.mp3': 1.0546459704023792,
         'disco.00057.mp3': 3.3724714044062325, 'metal.00081.mp3': 0.8753571640759281,
         'blues.00042.mp3': 0.9162412340798115, 'country.00058.mp3': 3.411828522656574,
         'jazz.00076.mp3': 3.412468610235584, 'metal.00017.mp3': 3.4070866449165385,
         'country.00099.mp3': 3.410349759697965, 'blues.00056.mp3': 0.38762883406070425,
         'classical.00002.mp3': -0.01225122635310625, 'disco.00016.mp3': -1.5877802896015203,
         'metal.00001.mp3': 1.0546459704023792, 'classical.00074.mp3': 0.49062196287304616,
         'reggae.00036.mp3': 0.38762883406070425, 'disco.00066.mp3': -4.027825577860107,
         'reggae.00053.mp3': 3.450123706124497, 'rock.00014.mp3': 1.0546459704023792,
         'disco.00007.mp3': -4.26039285128155, 'blues.00052.mp3': 0.4906219628730461,
         'hiphop.00028.mp3': 3.4174816710733076, 'disco.00048.mp3': 2.700569867600356,
         'reggae.00010.mp3': 1.0546459704023792, 'reggae.00091.mp3': 2.9390946636491586,
         'rock.00077.mp3': 3.408050002321556, 'blues.00009.mp3': 0.9659531077931642,
         'jazz.00065.mp3': 0.4906219628730461, 'blues.00041.mp3': 3.408050002321556, 'jazz.00088.mp3': 2.68902688735736,
         'classical.00073.mp3': 0.49062196287304616, 'pop.00007.mp3': 2.9390946636491586,
         'reggae.00023.mp3': 0.4906219628730461, 'reggae.00025.mp3': 1.0546459704023792,
         'blues.00078.mp3': -4.662803696203372, 'hiphop.00051.mp3': 3.4103497596979655,
         'reggae.00008.mp3': 3.4125287379551574, 'blues.00084.mp3': 3.408050002321556,
         'country.00046.mp3': 1.0546459704023785, 'disco.00076.mp3': 3.4121830166444314,
         'jazz.00005.mp3': 3.4103497596979655, 'country.00068.mp3': -1.9361620202499674,
         'metal.00039.mp3': 0.17957911577430946, 'pop.00025.mp3': 2.741121157843095,
         'disco.00024.mp3': 1.0546459704023792, 'blues.00045.mp3': -2.0603583870631312,
         'rock.00023.mp3': 3.407089249608243, 'classical.00072.mp3': 0.6230149291085746,
         'rock.00085.mp3': -5.042019322629769, 'hiphop.00042.mp3': 0.4899576012164488,
         'rock.00079.mp3': -1.3886868051826895, 'rock.00061.mp3': 1.0546459704023792,
         'classical.00077.mp3': -2.966936709742555, 'hiphop.00049.mp3': 2.192807993523653,
         'metal.00084.mp3': 0.6814999582874546, 'rock.00057.mp3': 0.4906219628730461,
         'disco.00036.mp3': 2.696842720233025, 'rock.00076.mp3': -4.621286412076003,
         'disco.00044.mp3': 4.258918271016281, 'disco.00032.mp3': -4.09380953928667,
         'metal.00036.mp3': 2.1827403653515822, 'blues.00064.mp3': 0.4906219628730461,
         'blues.00062.mp3': 2.7467080306625045, 'reggae.00076.mp3': -1.7898960623493725,
         'rock.00098.mp3': -4.3822557948947445, 'pop.00008.mp3': -0.6693716920568955,
         'classical.00022.mp3': 0.49062196287304616, 'hiphop.00036.mp3': -0.956151953921739,
         'classical.00000.mp3': 3.4502012133020954, 'rock.00006.mp3': 4.86298666235762,
         'pop.00010.mp3': -2.966936709742555, 'blues.00038.mp3': -3.8524597398492317,
         'hiphop.00055.mp3': 2.733948422648014, 'country.00080.mp3': 3.770677818577684,
         'metal.00069.mp3': 3.4118002265627263, 'country.00089.mp3': 4.257689232780961,
         'reggae.00068.mp3': 4.094392491268474, 'disco.00043.mp3': -1.0240218046863327,
         'reggae.00089.mp3': 0.4906219628730461, 'country.00040.mp3': 1.0546459704023785,
         'hiphop.00005.mp3': -3.0432736138720755, 'classical.00013.mp3': 0.49062196287304616,
         'jazz.00083.mp3': 4.862929361840961, 'reggae.00003.mp3': 3.407109786713278,
         'disco.00019.mp3': 3.410349759697965, 'rock.00059.mp3': -4.059672946015206,
         'metal.00011.mp3': -0.012251226353106373, 'reggae.00087.mp3': -1.501680460240311,
         'metal.00018.mp3': 3.4071719226001234, 'reggae.00005.mp3': 3.4103497596979655,
         'hiphop.00047.mp3': 2.6807239197876136, 'country.00001.mp3': 2.6849958588749483,
         'classical.00052.mp3': -0.7909936618954516, 'metal.00038.mp3': 3.1033686433527374,
         'classical.00017.mp3': -3.3753920964644415, 'pop.00075.mp3': -3.9744222606949178,
         'pop.00093.mp3': 3.412658148444356, 'country.00087.mp3': -0.012251226353106296,
         'blues.00028.mp3': 2.6849958588749483, 'rock.00074.mp3': -1.8585417218072708,
         'country.00014.mp3': 4.257665977169573, 'jazz.00040.mp3': 1.0546459704023792,
         'country.00010.mp3': 1.0546459704023792, 'reggae.00028.mp3': 0.425255070415186,
         'blues.00054.mp3': 3.4103497596979655, 'rock.00017.mp3': 3.7686086196245583,
         'metal.00026.mp3': -4.027825577860107, 'metal.00002.mp3': 1.0546459704023792,
         'jazz.00043.mp3': 3.7124017280843273, 'pop.00036.mp3': 1.9818774695378762,
         'rock.00084.mp3': -3.092646692556618, 'metal.00057.mp3': 0.7291979651802297,
         'metal.00091.mp3': 0.9403969890612313, 'metal.00040.mp3': -3.424367606385485,
         'jazz.00097.mp3': 0.3876288340607044, 'disco.00030.mp3': 3.410349759697965,
         'disco.00012.mp3': -1.4076119817863406, 'blues.00019.mp3': 3.411793260574253,
         'pop.00031.mp3': 3.0845961810246596, 'metal.00058.mp3': 0.8148403565688415,
         'country.00045.mp3': 1.0546459704023792, 'metal.00085.mp3': -1.4561209322407802,
         'country.00006.mp3': 3.4118658036514504, 'hiphop.00032.mp3': 2.689774664943406,
         'hiphop.00010.mp3': 2.741121157843095, 'jazz.00090.mp3': 3.4121830166444314,
         'disco.00084.mp3': 3.408050002321556, 'rock.00034.mp3': 0.6766454993438938, 'pop.00011.mp3': 3.408050002321556,
         'reggae.00093.mp3': 2.676766103849349, 'blues.00039.mp3': 4.863028896116814,
         'pop.00089.mp3': 3.7337486811401277, 'reggae.00037.mp3': 2.688997144173538,
         'disco.00022.mp3': 1.0546459704023785, 'rock.00082.mp3': -0.012251226353106288,
         'blues.00001.mp3': 0.4906219628730461, 'disco.00051.mp3': 3.4123144356586144,
         'rock.00064.mp3': 0.4906219628730461, 'hiphop.00081.mp3': 3.411843077755211,
         'metal.00034.mp3': -4.037624849335758, 'hiphop.00004.mp3': -1.6828200417947223,
         'classical.00016.mp3': 1.0546459704023792, 'jazz.00013.mp3': 1.7526449047182848,
         'classical.00029.mp3': 3.0860116267071214, 'reggae.00073.mp3': -1.0839479022751637,
         'blues.00014.mp3': 3.4501900980097164, 'country.00023.mp3': 3.4174816710733076,
         'metal.00082.mp3': 0.4899576012164488, 'blues.00005.mp3': 2.1841280537556265,
         'pop.00024.mp3': 1.0546459704023792, 'jazz.00024.mp3': 0.8916106125396389,
         'classical.00056.mp3': 0.9901617750425615, 'reggae.00030.mp3': 0.4906219628730461,
         'hiphop.00039.mp3': 3.4103497596979655, 'rock.00090.mp3': -3.796848996691655,
         'disco.00069.mp3': 1.0546459704023792, 'hiphop.00095.mp3': 1.288759049142816,
         'jazz.00074.mp3': 0.4906219628730461, 'country.00030.mp3': -3.843915090204101,
         'pop.00088.mp3': 0.3018503007563992, 'pop.00057.mp3': 3.410349759697965,
         'metal.00003.mp3': -0.10346602911753895, 'reggae.00060.mp3': 3.4071036718127017,
         'metal.00059.mp3': -1.5603741018084492, 'country.00042.mp3': 3.4121344588216833,
         'reggae.00049.mp3': 3.4103497596979655, 'country.00047.mp3': 0.490621962873045,
         'rock.00049.mp3': -5.57887837539346, 'disco.00008.mp3': 3.407094522046295, 'pop.00019.mp3': 3.4070860774325578,
         'reggae.00083.mp3': 1.0546459704023787, 'country.00034.mp3': 3.4122559622634046,
         'classical.00089.mp3': 5.447121252724237, 'hiphop.00003.mp3': 3.4103497596979655,
         'jazz.00081.mp3': 3.408050002321556, 'classical.00004.mp3': 4.258896781724414,
         'metal.00061.mp3': 2.6849958588749483, 'disco.00083.mp3': -0.06349441177540778,
         'rock.00038.mp3': 3.411793260574253, 'disco.00093.mp3': 0.540514166042988, 'jazz.00063.mp3': 4.257744824076412,
         'hiphop.00084.mp3': 2.884946526299006, 'reggae.00017.mp3': 3.4103497596979655,
         'jazz.00073.mp3': 0.4906219628730461, 'jazz.00067.mp3': -1.337999420955262,
         'blues.00098.mp3': 0.4906219628730461, 'disco.00026.mp3': 0.38762883406070436,
         'rock.00083.mp3': 1.0546459704023792, 'hiphop.00060.mp3': 3.4103497596979655,
         'rock.00046.mp3': 2.733948422648014, 'reggae.00096.mp3': 3.4103497596979655,
         'rock.00042.mp3': 3.4103497596979655, 'blues.00000.mp3': -3.292000804292834,
         'blues.00061.mp3': -0.6106779459431889, 'metal.00074.mp3': 0.4906219628730464,
         'pop.00064.mp3': 3.7337509680545002, 'classical.00015.mp3': 3.4121344588216833,
         'disco.00015.mp3': 2.0137671848964334, 'jazz.00034.mp3': 3.3724714044062325,
         'rock.00010.mp3': -1.550381432777417, 'disco.00094.mp3': 0.49062196287304494,
         'country.00094.mp3': 0.490621962873045, 'metal.00014.mp3': -0.7067748277332901,
         'pop.00035.mp3': 3.4122559622634046, 'metal.00080.mp3': 4.091253083702792,
         'metal.00055.mp3': 1.3519792481562707, 'blues.00063.mp3': 4.863010975305086,
         'jazz.00060.mp3': 2.192807993523653, 'pop.00020.mp3': 0.49062196287304494,
         'hiphop.00073.mp3': 3.611727972936322, 'country.00059.mp3': -0.6739790721240739,
         'blues.00075.mp3': -0.2620538229227109, 'reggae.00061.mp3': -2.1094588759990347,
         'disco.00034.mp3': 3.408050002321556, 'jazz.00025.mp3': 4.257721764887122,
         'reggae.00048.mp3': 3.4117932585154382, 'rock.00015.mp3': 3.410349759697965,
         'rock.00032.mp3': 3.408050002321556, 'blues.00013.mp3': 1.0546459704023792,
         'hiphop.00009.mp3': -2.0027576704717434, 'blues.00006.mp3': 0.8578220823776829,
         'metal.00004.mp3': 2.6908864895107625, 'classical.00097.mp3': 3.4121344588216833,
         'blues.00030.mp3': 1.0546459704023792, 'country.00071.mp3': -0.4239024813901552,
         'hiphop.00007.mp3': -4.089669181115351, 'country.00035.mp3': 3.410349759697965,
         'hiphop.00002.mp3': -0.4710578874730886, 'pop.00056.mp3': 3.4332994113229427,
         'country.00028.mp3': 4.258906598476537, 'rock.00043.mp3': 3.412086847042682,
         'metal.00060.mp3': -0.2638156703605718, 'country.00070.mp3': 3.410349759697965,
         'disco.00099.mp3': 3.408050002321556, 'country.00019.mp3': 3.0847262529721093,
         'jazz.00018.mp3': 3.408050002321556, 'hiphop.00059.mp3': 3.4070860774325578,
         'jazz.00004.mp3': -0.05074345401003491, 'hiphop.00006.mp3': 4.094392228786266,
         'jazz.00075.mp3': 2.733948422648014, 'rock.00002.mp3': -3.3731112322330428,
         'blues.00040.mp3': 3.410349759697965, 'country.00024.mp3': 0.490621962873045,
         'jazz.00017.mp3': 3.103733748805397, 'classical.00005.mp3': 0.49062196287304616,
         'blues.00079.mp3': -0.19655654599045616, 'blues.00080.mp3': -0.6259442086353048,
         'metal.00095.mp3': -2.67694541285599, 'classical.00009.mp3': 3.4103497596979655,
         'pop.00045.mp3': 2.733948422648014, 'classical.00040.mp3': 3.4214295687333505,
         'rock.00097.mp3': 1.0546459704023792, 'metal.00015.mp3': 2.6813429023518154,
         'blues.00068.mp3': 0.4899576012164488, 'classical.00091.mp3': -1.9361620202499672,
         'rock.00099.mp3': -5.38880419430043, 'disco.00074.mp3': 1.4751517276910366,
         'country.00013.mp3': 3.410349759697965, 'hiphop.00019.mp3': 2.1827403653515822,
         'pop.00051.mp3': 1.0546459704023792, 'classical.00039.mp3': 3.7337509680545002,
         'pop.00081.mp3': -0.012251226353106283, 'jazz.00008.mp3': 3.4103497596979655,
         'metal.00066.mp3': -1.8587709985836314, 'classical.00070.mp3': 0.6988948688928155,
         'blues.00087.mp3': 3.410349759697965, 'reggae.00040.mp3': 3.733711100319032,
         'blues.00026.mp3': 3.0853173898779795, 'metal.00050.mp3': 2.733948422648014,
         'rock.00055.mp3': 0.23206081068904835, 'disco.00055.mp3': 5.447121252724237,
         'jazz.00059.mp3': 3.321914497043449, 'rock.00078.mp3': -4.321493582624374, 'pop.00065.mp3': 3.424459232507422,
         'country.00067.mp3': 0.490621962873045, 'hiphop.00008.mp3': 2.8545990545241966,
         'rock.00080.mp3': -0.7979003575513385, 'blues.00025.mp3': -1.713334043458703,
         'classical.00045.mp3': 3.412468610235584, 'jazz.00050.mp3': 2.057727318440142,
         'metal.00087.mp3': -0.5009956691645988, 'reggae.00057.mp3': 3.0853173898779795,
         'metal.00068.mp3': -2.966936709742555, 'country.00036.mp3': 3.408050002321556,
         'disco.00059.mp3': 2.9390946636491586, 'jazz.00078.mp3': 3.4073016916017447,
         'metal.00020.mp3': 2.7361687118663673, 'hiphop.00074.mp3': 1.3648382795868141,
         'reggae.00021.mp3': -2.1319210152475008, 'disco.00090.mp3': 4.25891205548629,
         'metal.00019.mp3': -0.012251226353106373, 'reggae.00042.mp3': -0.750614280497586,
         'country.00003.mp3': 2.6765449349015613, 'blues.00049.mp3': -3.3753920964644415,
         'blues.00055.mp3': 0.03778355112777819, 'disco.00039.mp3': -1.491104996652523,
         'blues.00072.mp3': 0.5049759336056764, 'blues.00044.mp3': -3.5290556832632993,
         'hiphop.00017.mp3': 1.0546459704023792, 'country.00081.mp3': 5.406902713593346,
         'pop.00038.mp3': -3.3753920964644415, 'metal.00008.mp3': 3.408050002321556,
         'country.00044.mp3': 3.4121830166444314, 'rock.00075.mp3': -3.23817694122689,
         'disco.00040.mp3': 2.8836191181845416, 'hiphop.00082.mp3': 3.7686099977853433,
         'jazz.00022.mp3': 3.4502012133020954, 'classical.00084.mp3': 3.4103497596979655,
         'country.00085.mp3': 3.4502182163212005, 'disco.00082.mp3': 0.8056993766376698,
         'reggae.00004.mp3': 3.085310447102329, 'country.00090.mp3': 3.4214164939903804,
         'metal.00046.mp3': 3.4103497596979655, 'classical.00090.mp3': 2.2373310290460275,
         'pop.00073.mp3': 3.408050002321556, 'pop.00050.mp3': 3.411789030611519, 'hiphop.00078.mp3': 5.447110566628485,
         'classical.00028.mp3': 2.1832048246865963, 'jazz.00009.mp3': -1.7832448116233426,
         'pop.00055.mp3': 3.408050002321556, 'metal.00067.mp3': -0.38639458011804,
         'hiphop.00054.mp3': 2.430193094451623, 'metal.00045.mp3': -1.837556044118872,
         'hiphop.00040.mp3': -0.3739019149896516, 'blues.00027.mp3': 0.4906219628730461,
         'country.00011.mp3': 1.9722126313509583, 'pop.00067.mp3': 0.8899323381154574,
         'jazz.00010.mp3': 1.9722126313510138, 'classical.00021.mp3': 3.411798255886065,
         'reggae.00020.mp3': 0.8916106125396389, 'pop.00091.mp3': 3.4213909905624025,
         'country.00032.mp3': 3.0852373822834025, 'metal.00064.mp3': 2.7361687118663673,
         'disco.00091.mp3': 1.0546459704023785, 'disco.00017.mp3': 3.411793260574253,
         'reggae.00071.mp3': 2.68902688735736, 'pop.00058.mp3': 3.611727972936322,
         'rock.00054.mp3': 0.16611550592590488, 'hiphop.00048.mp3': 3.4103497596979655,
         'country.00004.mp3': 0.3876288340607044, 'metal.00029.mp3': -0.7766249409438709,
         'jazz.00003.mp3': 3.4103497596979655, 'disco.00047.mp3': 0.49062196287304494,
         'pop.00086.mp3': 3.76866510956774, 'jazz.00030.mp3': -0.7658850586205097,
         'classical.00019.mp3': 1.9564983430273624, 'metal.00024.mp3': 0.9659531077931642,
         'disco.00033.mp3': -2.120997396221237, 'country.00020.mp3': 3.4096598716879787,
         'metal.00054.mp3': -4.020310537978787, 'metal.00070.mp3': 5.447110566628485,
         'reggae.00065.mp3': 3.407170945027772, 'blues.00012.mp3': 2.733948422648014}
    b = []
    for i in a.values():
        b.append(i)
    print(max(b), min(b))
    plt.hist(b, bins=np.arange(-6, 7, 0.5))
    plt.title("histogram")
    plt.show()

    c = {'A2PQHWG79HDCX6': 4.43725244418267, 'A1MGP6RT6R6HS7': 0.25923042823676556,
         'A28G4QO0DRY8OZ': 3.2784156907648447, 'AZ9A137H25EDX': 6.205443248142304, 'A1N274L13ZKKI5': 14.926028227502314,
         'A1OB2UYMHESJIQ': 0.4677810611337952, 'AK0F484W5TBJ4': -2.243825597036225, 'A12YUYYSIZ8GT2': 4.020616724536333,
         'A3NUWV8UAHEDQ1': 6.2031885141883185, 'A32CMQGGTSUK6Z': 0.020257968113225326,
         'A2D4GSZPZB56HW': 0.6137640724225287, 'AEK67RC8DN148': 3.3089141316568513,
         'A175ZTL2OZO9YU': 0.08856766540616054, 'A646R8SV0S04Y': 0.020939496566417033,
         'AC6QZ6VSEIYK1': 0.8329277302009501, 'A242LCRYG0BH30': -6.503612548573747, 'AZJEGHE4605S5': 7.435385026004251,
         'A1ITBXITLY2952': 2.88706502706714, 'A2PHDF44BN2OGZ': 0.9702155664031153, 'ANCNIZC2AB5Z8': 0.366531760215723,
         'A2H5YK4RMOE7UI': 1.273679125729652, 'AL12RG9EJZ60': -0.16823184293011684, 'AQEVAKQLI09EN': 1.5001901456214821,
         'A32L9ALQMIRPQW': 0.4599950187118553, 'A2SMDHLM9CQJWM': 3.8450433675417477,
         'A1GEAO2U4ZR253': 4.2077605947005745, 'A3AWC4P8QUK1XB': 0.32176309857111807,
         'A3G4O512F59PH0': 0.45202567740031735, 'A3QIEF1HBF69Y4': 3.8100159035919328,
         'A7NZ814Q5MX2R': 2.4787009974344403, 'ARX0S1CIDJLOX': 1.2690987637512587, 'A3FNONESNZKWVB': 4.3947212232475,
         'AKJWYGT4NN2S1': 0.0265424225763751, 'A3HLZQNQ1O5SMC': 8.280966310743624, 'A3546X291KZ1PV': 2.5023572346623317,
         'A14XY4KSUATVAA': 0.011652297781223555, 'A28L1K6D8QUCML': 0.47862779168762,
         'A1FUBGE9CHDHU6': 0.06541471468523088, 'A292QSLC0BUT0O': 1.7677718906274542,
         'A4SDBFRJ557H4': -0.03383026343797714, 'AZT6P0RS7ZHCV': 1.2025246390835391, 'A1F4D2PZ7NNWTL': 4.80757145382033,
         'AA3V5BYE3MSQW': 0.11801728608642403, 'A1G05O3HM7DNVZ': 4.075304468408828}
    d = []
    for i in c.values():
        d.append(i)
    print(max(d), min(d))
    plt.hist(d, bins=np.arange(-7, 15, 0.5))
    plt.title("histogram")
    plt.show()


def acc_row():
    c = []
    d = []
    path = 'C:\\Users\\isaac\\Desktop\\testglad'
    colors = ['k', 'y', 'g', 'r']
    for index, i in enumerate([2, 5]):
        a = []
        b = []
        filepath = os.path.join(path, 'beta' + str(i))
        with open(filepath) as f:
            for j in f.readlines():
                tem = j.strip().split()
                if not tem:
                    continue
                a.append(float(tem[0][10:]))
                b.append(float(tem[1][9:]))
        x_axes = range(len(a))
        plt.plot(x_axes, a, c=colors[index], linewidth=0.3)
        plt.plot(x_axes, b, c=colors[index], linewidth=0.3)
    plt.grid(b=True)
    plt.show()


def check_worker_10():
    w_1 = np.load('C:\\Users\\isaac\\Desktop\\worker_hun.npy')
    b = []
    for index, i in enumerate(w_1):
        b.append((index, np.sum(i)))
    b = sorted(b, key=lambda x: x[1], reverse=True)
    for i in b:
        print(i)


def pear():
    w_2 = np.load(r'D:\aaai\10分类出图\ds\all_worker_l1.npy')[-1]
    w_acc = np.load(r'D:\aaai\worker_acc.npy')[()]
    w_alpha = np.load(r'C:\Users\isaac\Desktop\all_worker_alpha.npy')[-1]
    workers = np.load(r'D:\aaai\music_data.npz')['workers'].tolist()
    worker_number = len(workers)
    x = []
    y = []
    z = []
    for i in range(worker_number):
        wname = workers[i]
        x.append(w_acc[wname][0])
        y.append(np.trace(w_2[i]) / 10)
        z.append(w_alpha[i])
    print(pearsonr(x, y))
    print(pearsonr(x, z))

    w_2 = np.load('D:\\aaai\\2分类出图\\ds\\all_worker_l1.npy')[-1]
    w_acc = np.load('D:\\aaai\\worker_acc_2.npy')[()]
    w_alpha = np.load('C:\\Users\\isaac\\Desktop\\all_worker_alpha_2.npy')[2885]
    workers = np.load('D:\\aaai\\workers_name.npz')['workers'].tolist()
    worker_number = len(workers)
    x = []
    y = []
    z = []
    for i in range(worker_number):
        wname = workers[i]
        x.append(w_acc[wname][0])
        y.append(np.trace(w_2[i]) / 10)
        z.append(w_alpha[i])
    print(pearsonr(x, y))
    print(pearsonr(x, z))


def simple():
    """less是轮数少的意思"""
    x1 = range(2001)
    y1 = np.load('D:\\aaai\\2_class_glad_less\\0\\all_train_acc.npy')[()]
    print(y1.shape)
    x2 = range(2001)
    y2 = np.load(r'D:\aaai\2_class_glad_less\0\all_test_acc.npy')[()]
    x3 = range(2001)
    y3 = np.load(r'D:\aaai\2_class_glad_less\0\beta700.npy')[()]
    print(y3.shape)
    plt.figure(figsize=(10, 7), dpi=200)
    plt.plot(x1, y1, linewidth=0.5)
    plt.plot(x2, y2, linewidth=0.5)
    # plt.plot(x3, y3, linewidth=0.5)
    plt.grid()
    # 保存(svg文件在浏览器中打开放大不会失帧)
    # 展示图形
    plt.show()


def timedraw2():
    """测试agg和我们的时间，2分类，e1和e2是时间，e1是agg"""
    e1 = 1803
    x1 = [(e1 / 200) * i for i in range(200)]
    y1 = np.load(r'D:\aaai\rebuttal测试agg和我们的时间\2\agg\train_acc.npy')

    e2 = 858
    x2 = [(e2 / 200) * i for i in range(200)]
    y2 = np.load(r'D:\aaai\rebuttal测试agg和我们的时间\2\our\all_train_acc.npy')

    plt.figure(figsize=(10, 7), dpi=200)

    # 绘制折线图
    plt.plot(x1, y1, linewidth=0.5)
    plt.plot(x2, y2, linewidth=0.5)
    plt.xticks(range(0, 1900, 100), fontsize=5)
    plt.yticks([x / 100 for x in range(0, 100, 5)])
    plt.grid()
    plt.show()


def timedraw10():
    """测试agg和我们的时间，10分类，e1和e2是时间，e1是agg"""
    e1 = 2901
    x1 = [(e1 / 4000) * i for i in range(4000)]
    y1 = np.load(r'D:\aaai\rebuttal测试agg和我们的时间\10\agg\train_acc.npy')

    e2 = 1314
    x2 = [(e2 / 10000) * i for i in range(10000)]
    y2 = np.load(r'D:\aaai\rebuttal测试agg和我们的时间\10\our\all_train_acc.npy')

    plt.figure(figsize=(10, 7), dpi=200)

    # 绘制折线图
    plt.plot(x1, y1, linewidth=0.3)
    plt.plot(x2, y2, linewidth=0.3)
    plt.xticks(range(0, 3000, 50), fontsize=3)
    plt.yticks([x / 100 for x in range(0, 80, 2)])
    plt.grid()
    plt.show()


def func3(x):
    return -3.002 * x * x * x + 5.389 * x * x - 1.658 * x + 0.3572


def func2_qian(x):
    return -1.227 * x * x * x + 1.411 * x * x + 1.177 * x - 0.2852


def func2_tanh(x):
    return (np.tanh((x - 0.81) * 20) + 1.05) / 2

def func10_qian(x):
    return 1.067 * x - 0.01234


def func10_tanh(x):
    return (np.tanh((x - 0.74) * 20) * 1.14 + 0.89) / 2


def sandian():
    w_2 = np.load('D:\\aaai\\12-30\\2c\\ds\\results1\\1\\all_worker_l1.npy')
    has_nan = [np.isnan(x).any() for x in w_2]
    if True in has_nan:
        w_2_index = has_nan.index(True) - 1
    else:
        w_2_index = -1
    print(w_2_index)
    w_2 = w_2[w_2_index]
    w_acc = np.load('D:\\aaai\\worker_acc_2.npy')[()]
    workers = np.load('D:\\aaai\\workers_name.npz')['workers'].tolist()
    worker_alpha_dan = np.load('D:\\aaai\\12-30\\2c\\glad\\results1\\1\\all_worker_alpha.npy')
    has_nan = [np.isnan(x).any() for x in worker_alpha_dan]
    if True in has_nan:
        w_dan_index = has_nan.index(True) - 1
    else:
        w_dan_index = -1
    print(w_dan_index)
    worker_alpha_dan = worker_alpha_dan[w_dan_index]

    w_acc_sort = sorted(w_acc.items(), key=lambda x: x[1][1], reverse=True)
    print(w_acc_sort)

    x = []
    y = []
    z = []
    for i in range(20):
        wname = w_acc_sort[i][0]
        windex = workers.index(wname)
        x.append(w_acc[wname][0])
        y.append(np.trace(w_2[windex]) / 2)
        z.append(np.tanh(worker_alpha_dan[windex]))
    print(pearsonr(x, y))
    print(pearsonr(x, z))

    plt.figure(figsize=(10, 10.1))
    plt.subplot2grid((2, 2), (0, 0))
    plt.xlim(0.4, 1.03)
    plt.ylim(0.4, 1.05)
    plt.xlabel('True ability', fontsize=16)
    plt.ylabel('Estimated ability', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.scatter(x, y, s=18)

    x_1 = [0.48654104979811574, 0.6983050847457627, 0.7614942528735632, 0.8194726166328601]
    y_1 = [0.4804268479347229, 0.807377815246582, 0.8879629969596863, 0.9692994356155396]

    print(sorted(zip(x, y), key=lambda x: x[0]))
    x_2 = [0.7614942528735632, 0.8194726166328601, 0.8885245901639345, 0.903448275862069, 0.9273570324574961, 0.946067415730337]
    y_2 = [0.8879629969596863, 0.9692994356155396, 0.9917293787002563, 0.99509596824646, 0.9989581108093262, 1]

    f1 = np.polyfit(x_1, y_1, 3)
    p1 = np.poly1d(f1)
    print(p1)
    y_1 = p1(x_1)
    plt.plot(x_1, y_1, linewidth=2)

    f2 = np.polyfit(x_2, y_2, 3)
    p2 = np.poly1d(f2)
    print(p2)
    x_2 = [x / 100 for x in range(82, 96, 2)]
    y_2 = p2(x_2)
    plt.plot(x_2, y_2, c='C0', linewidth=2)

    ax = plt.subplot2grid((2, 2), (0, 1))
    xz = sorted(zip(x, z), key=lambda ij: ij[0])
    x_shun = []
    z_shun = []
    for i in range(50, 96, 2):
        x_shun.append(i / 100)
        z_shun.append(func2_tanh(i / 100))
    plt.plot(x_shun, z_shun, linewidth=2)
    x.pop(1)
    z.pop(1)
    plt.xlabel('True ability', fontsize=16)
    plt.ylabel('Estimated ability', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.scatter(x, z, s=18)
    plt.scatter([0.48654104979811574, ], [-0.010108414, ], c='r', s=18)
    ax.annotate('(0.4865, -0.0101)', xy=(0.486, 0.02), xytext=(0.486, 0.26), fontsize=14,
                arrowprops=dict(width=1.5, headwidth=7, facecolor='black', shrink=0.01))


    w_2 = np.load('D:\\aaai\\12-30\\10c\\ds\\results1\\1\\all_worker_l1.npy')
    has_nan = [np.isnan(x).any() for x in w_2]
    if True in has_nan:
        w_2_index = has_nan.index(True) - 1
    else:
        w_2_index = -1
    print(w_2_index)
    w_2 = w_2[w_2_index]
    w_acc = np.load('D:\\aaai\\worker_acc.npy')[()]
    workers = np.load('D:\\aaai\\music_data.npz')['workers'].tolist()
    worker_alpha_dan = np.load('D:\\aaai\\12-30\\10c\\glad\\results1\\1\\all_worker_alpha.npy')
    has_nan = [np.isnan(x).any() for x in worker_alpha_dan]
    if True in has_nan:
        w_dan_index = has_nan.index(True) - 1
    else:
        w_dan_index = -1
    print(w_dan_index)
    worker_alpha_dan = worker_alpha_dan[w_dan_index]

    w_acc_sort = sorted(w_acc.items(), key=lambda x: x[1][1], reverse=True)
    print(w_acc_sort)

    x = []
    y = []
    z = []
    for i in range(20):
        wname = w_acc_sort[i][0]
        windex = workers.index(wname)
        x.append(w_acc[wname][0])
        y.append(np.trace(w_2[windex]) / 10)
        z.append(np.tanh(worker_alpha_dan[windex]))
    print(pearsonr(x, y))
    print(pearsonr(x, z))
    print(sorted(zip(x, y), key=lambda x: x[0]))

    plt.subplot2grid((2, 2), (1, 0))
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.xlabel('True ability', fontsize=16)
    plt.ylabel('Estimated ability', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.scatter(x, y, s=18)
    x_1 = [x / 100 for x in range(4, 100, 6)]
    plt.plot(x_1, [func10_qian(x) for x in x_1], linewidth=2)

    ax = plt.subplot2grid((2, 2), (1, 1))
    xz = sorted(zip(x, z), key=lambda ij: ij[0])
    print(xz)
    x_shun = []
    z_shun = []
    for i in range(10, 96, 2):
        x_shun.append(i / 100)
        z_shun.append(func10_tanh(i / 100))
    plt.plot(x_shun, z_shun, linewidth=2)
    x.pop(1)
    z.pop(1)
    plt.xlabel('True ability', fontsize=16)
    plt.ylabel('Estimated ability', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.scatter(x, z, s=18)
    plt.scatter([0.06804733727810651, ], [-0.19357558, ], c='r', s=18)
    ax.annotate('(0.068, -0.193)', xy=(0.068, -0.15), xytext=(0.068, 0.16), fontsize=14,
                arrowprops=dict(width=1.5, headwidth=7, facecolor='black', shrink=0.01))

    plt.figtext(0.26, 0.525, '(a) Model: SpeeLFC; Dataset: SPC', va="center", ha="center", size=18)
    plt.figtext(0.77, 0.525, '(b) Model: SpeeLFC-D; Dataset: SPC', va="center", ha="center", size=18)
    plt.figtext(0.26, 0.02, '(c) Model: SpeeLFC; Dataset: MGC', va="center", ha="center", size=18)
    plt.figtext(0.77, 0.02, '(d) Model: SpeeLFC-D; Dataset: MGC', va="center", ha="center", size=18)
    plt.subplots_adjust(left=0.08, bottom=0.1, top=0.99, right=0.97, hspace=0.3, wspace=0.3)
    plt.savefig('somt.pdf', format='pdf')


def 做题最多的6个人2分类():
    w_1 = np.load('D:\\aaai\\worker_hun_2.npy')
    w_2 = np.load('D:\\aaai\\12-30\\2c\\ds\\results1\\1\\all_worker_l1.npy')[-1]
    w_acc = np.load('D:\\aaai\\worker_acc_2.npy')[()]
    workers = np.load('D:\\aaai\\workers_name.npz')['workers'].tolist()

    c_w = np.load('D:\\aaai\\crowdlayer_weights_2.npy')[0]
    worker_number = len(workers)
    w_acc_sort = sorted(w_acc.items(), key=lambda xx: xx[1][1], reverse=True)
    print(w_acc_sort)

    chose_id = []
    for index, i in enumerate(w_acc_sort):
        chose_id.append(workers.index(i[0]))
        if index == 5:
            break
    print(chose_id)
    real_worker = []
    learn_worker = []
    crowd_worker = []
    wids = []
    wn = []
    wr = []
    for i in chose_id:
        wid = i
        wname = workers[wid]
        wids.append(wid)
        this_true = w_1[wid]
        zero_index = []
        for index_j, j in enumerate(this_true):
            if sum(j) == 0:
                zero_index.append(index_j)
        for j in zero_index:
            this_true[j] = [1, ] * worker_number
        temp_y = c_w[:, :, wid]
        # temp_y = temp_y + np.abs(c_w.min())
        # crowd_worker.append(temp_y / temp_y.sum(axis=0))
        crowd_worker.append(temp_y)
        real_worker.append(this_true)
        learn_worker.append(w_2[wid])
        wn.append(w_acc[wname][1])
        wr.append(round(w_acc[wname][0], 2))
    compare_conf_mats_2(2, real_worker, learn_worker, crowd_worker, wids, wn, wr, [])


def 做题最多的6个人10分类():
    w_1 = np.load('D:\\aaai\\worker_hun.npy')
    w_2 = np.load('D:\\aaai\\12-30\\10c\\ds\\results1\\1\\all_worker_l1.npy')[-1]
    w_acc = np.load('D:\\aaai\\worker_acc.npy')[()]
    workers = np.load('D:\\aaai\\music_data.npz')['workers'].tolist()

    worker_number = len(workers)
    w_acc_sort = sorted(w_acc.items(), key=lambda xx: xx[1][1], reverse=True)
    print(w_acc_sort)

    chose_id = []
    for index, i in enumerate(w_acc_sort):
        chose_id.append(workers.index(i[0]))
        if index == 5:
            break
    real_worker = []
    learn_worker = []
    crowd_worker = []
    wids = []
    wn = []
    wr = []
    for i in chose_id:
        wid = i
        wname = workers[wid]
        wids.append(wid)
        this_true = w_1[wid]
        zero_index = []
        for index_j, j in enumerate(this_true):
            if sum(j) == 0:
                zero_index.append(index_j)
        for j in zero_index:
            this_true[j] = [1, ] * worker_number
        real_worker.append(this_true)
        learn_worker.append(w_2[wid])
        wn.append(w_acc[wname][1])
        wr.append(round(w_acc[wname][0], 2))
    compare_conf_mats_10(10, real_worker, learn_worker, crowd_worker, wids, wn, wr, [])


if __name__ == '__main__':
    做题最多的6个人2分类()
