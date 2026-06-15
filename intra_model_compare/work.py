import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from pathlib import Path


def load_csv(file_path):

    table = []
    with open(file_path, encoding='utf-8-sig') as fd:
        for record in csv.DictReader(fd):
            table.append(record)

    return table


def draw(content, columns, cmp_name, save_path, metric_label):
    x_c, y_c = columns

    models = sorted(list(set([
        i['model_name']
        for i in content
    ])))

    plt.figure(figsize=(8, 6))

    color_list = ['#F8766D', '#619CFF', '#111111']
    marker = ['o', 's', '^']

    # Create scatter plot
    for m, color, ma in zip(models, color_list, marker):
        x = [
            float(i[x_c]) * 100
            for i in content
            if i['model_name'] == m
        ]
        y = [
            float(i[y_c]) * 100
            for i in content
            if i['model_name'] == m
        ]
        plt.scatter(x, y, color=color, label=m, s=20, marker=ma)

    plt.plot([0, 110], [0, 110], '--', color='gray')

    # Add labels and title
    plt.xlabel('Base (%)', fontweight='bold', fontsize=18)
    plt.ylabel(f'{cmp_name} (%)', fontweight='bold', fontsize=18)
    # plt.title(Path(save_path).stem, fontsize=27, pad=10)

    plt.legend(fontsize=16)

    ticks = [0, 25, 50, 75, 100]
    plt.xticks(ticks, fontsize=16)
    plt.yticks(ticks, fontsize=16)

    ax = plt.gca()
    ax.set_xticklabels(['0', '25', '50', '75', '100'])
    ax.set_yticklabels(['', '25', '50', '75', '100'])
    ax.text(
        0.98,
        0.02,
        metric_label,
        transform=ax.transAxes,
        ha='right',
        va='bottom',
        fontsize=22,
        fontweight='bold',
        bbox={'facecolor': 'white', 'alpha': 0.8, 'edgecolor': 'none', 'pad': 4}
    )

    plt.xlim(0, 110)
    plt.ylim(0, 110)

    plt.grid(True, linestyle='-', alpha=0.3)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def compose_grid(image_paths, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for ax, image_path in zip(axes.flat, image_paths):
        ax.imshow(mpimg.imread(image_path))
        ax.axis('off')

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def work():
    content = load_csv('./evaluation_metrics_by_qid_full150_converted.csv')
    # print(content[0])
    ft_accuracy_path = 'FT/FT_Accuracy.tiff'
    ft_precision_path = 'FT/FT_Precision.tiff'
    ft_recall_path = 'FT/FT_Recall.tiff'
    ft_f1_path = 'FT/FT_F1.tiff'

    draw(content, ['base A', 'FT A'], 'FT', ft_accuracy_path, 'Accuracy')
    draw(content, ['base P', 'FT P'], 'FT', ft_precision_path, 'Precision')
    draw(content, ['base R', 'FT R'], 'FT', ft_recall_path, 'Recall')
    draw(content, ['base F', 'FT F'], 'FT', ft_f1_path, 'F1-Score')
    compose_grid(
        [ft_accuracy_path, ft_precision_path, ft_recall_path, ft_f1_path],
        'FT/FT_2x2.tiff'
    )

    qsp_accuracy_path = 'QSP/QSP_Accuracy.tiff'
    qsp_precision_path = 'QSP/QSP_Precision.tiff'
    qsp_recall_path = 'QSP/QSP_Recall.tiff'
    qsp_f1_path = 'QSP/QSP_F1.tiff'

    draw(content, ['base A', 'QSP A'], 'QSP', qsp_accuracy_path, 'Accuracy')
    draw(content, ['base P', 'QSP P'], 'QSP', qsp_precision_path, 'Precision')
    draw(content, ['base R', 'QSP R'], 'QSP', qsp_recall_path, 'Recall')
    draw(content, ['base F', 'QSP F'], 'QSP', qsp_f1_path, 'F1-Score')
    compose_grid(
        [qsp_accuracy_path, qsp_precision_path, qsp_recall_path, qsp_f1_path],
        'QSP/QSP_2x2.tiff'
    )

    draw(content, ['base A', 'FT+QSP A'], 'FT+QSP', 'FT_QSP/FT_QSP_Accuracy.tiff', 'Accuracy')
    draw(content, ['base P', 'FT+QSP P'], 'FT+QSP', 'FT_QSP/FT_QSP_Precision.tiff', 'Precision')
    draw(content, ['base R', 'FT+QSP R'], 'FT+QSP', 'FT_QSP/FT_QSP_Recall.tiff', 'Recall')
    draw(content, ['base F', 'FT+QSP F'], 'FT+QSP', 'FT_QSP/FT_QSP_F1.tiff', 'F1-Score')


if __name__ == '__main__':
    work()
