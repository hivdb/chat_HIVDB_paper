import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_csv(file_path):

    table = []
    with open(file_path, encoding='utf-8-sig') as fd:
        for record in csv.DictReader(fd):
            table.append(record)

    return table


def draw(content, columns, cmp_name, save_path):
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

    plt.legend()

    ticks = [0, 25, 50, 75, 100]
    plt.xticks(ticks)
    plt.yticks(ticks)

    ax = plt.gca()
    ax.set_xticklabels(['0', '25', '50', '75', '100'])
    ax.set_yticklabels(['', '25', '50', '75', '100'])

    plt.xlim(0, 110)
    plt.ylim(0, 110)

    plt.grid(True, linestyle='-', alpha=0.3)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)


def work():
    content = load_csv('./evaluation_metrics_by_qid_full150_converted.csv')
    # print(content[0])
    draw(content, ['base A', 'FT A'], 'FT', 'FT/FT_Accuracy.svg')
    draw(content, ['base P', 'FT P'], 'FT', 'FT/FT_Precision.svg')
    draw(content, ['base R', 'FT R'], 'FT', 'FT/FT_Recall.svg')
    draw(content, ['base F', 'FT F'], 'FT', 'FT/FT_F1.svg')

    draw(content, ['base A', 'QSP A'], 'QSP', 'QSP/QSP_Accuracy.svg')
    draw(content, ['base P', 'QSP P'], 'QSP', 'QSP/QSP_Precision.svg')
    draw(content, ['base R', 'QSP R'], 'QSP', 'QSP/QSP_Recall.svg')
    draw(content, ['base F', 'QSP F'], 'QSP', 'QSP/QSP_F1.svg')

    draw(content, ['base A', 'FT+QSP A'], 'FT+QSP', 'FT_QSP/FT_QSP_Accuracy.svg')
    draw(content, ['base P', 'FT+QSP P'], 'FT+QSP', 'FT_QSP/FT_QSP_Precision.svg')
    draw(content, ['base R', 'FT+QSP R'], 'FT+QSP', 'FT_QSP/FT_QSP_Recall.svg')
    draw(content, ['base F', 'FT+QSP F'], 'FT+QSP', 'FT_QSP/FT_QSP_F1.svg')


if __name__ == '__main__':
    work()
