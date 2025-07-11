from collections import Counter

from reorganization_verifier.accuracy_verifier.section_comparer import get_corrections
from util import helper
from util.constants import section_subsection_headers


def print_section_subsection_table(mismatched_sections) -> None:
    sorted_mismatched_sections = dict(sorted(mismatched_sections.items(), key=lambda item: len(item[1]), reverse=True))

    print('\t\\begin{tabular}{llr}')
    print('\t\t\\toprule')
    print('\t\t\\textbf{Sections} & \\textbf{Subsections} & \\textbf{\# of Cards} \\\\')
    print('\t\t\midrule')
    for section_header, mismatched_subsections in sorted_mismatched_sections.items():
        if len(sorted_mismatched_sections[section_header]) == 0:
            break
        freq = Counter(mismatched_subsections)
        if section_header in freq:
            display_header = section_header.replace('#', '').replace(':', '').strip()
            print(f'\t\t{display_header} & & {freq[section_header]} \\\\')
        else:
            section_subheaders_with_correction = list(
                set(freq.keys()) & set(section_subsection_headers[section_header]))
            section_freq = {
                k: v for k, v in freq.items()
                if any(substring in k for substring in section_subheaders_with_correction)
            }
            sorted_dict = dict(sorted(section_freq.items(), key=lambda item: item[1], reverse=True))
            print(
                f'\t\t\multirow{{{len(sorted_dict) + 1}}}{{*}}{{{section_header.replace("#", "").replace(":", "").strip()}}}')
            for subsection_header, subsection_freq in sorted_dict.items():
                display_header = subsection_header.replace('#', '').replace(':', '').strip()
                print(f'\t\t & {display_header} & {subsection_freq} \\\\')
            print('\t\t\cmidrule(lr){2-3}')
            print(f'\t\t & & {len(mismatched_subsections)} \\\\')
        print('\t\t\midrule')
    print(
        f'\t\t\\textbf{{Total}} & \multicolumn{{2}}{{r}}{{\\textbf{{{sum(len(v) for v in mismatched_sections.values())} out of {sum(len(v) for v in section_subsection_headers.values()) * 48}}}}} \\\\')
    print('\t\t\\bottomrule')
    print('\t\end{tabular}')


def print_subsection_table(mismatched_sections):
    pairs = [(section, sub) for section, subs in mismatched_sections.items() for sub in subs]

    pair_counts = Counter(pairs)
    sorted_counts = pair_counts.most_common()

    print('\t\\begin{tabular}{lr}')
    print('\t\t\\toprule')
    print('\t\t\\textbf{Subsections} & \\textbf{\# of Cards} \\\\')
    print('\t\t\midrule')

    for (section, subsection), count in sorted_counts:
        display_section = section.replace('#', '').replace(':', '').strip()
        display_subsection = subsection.replace('#', '').replace(':', '').strip()

        if display_section == display_subsection:
            print(f'\t\t{display_section} & {count} \\\\')
        else:
            print(f'\t\t{display_section}/{display_subsection} & {count} \\\\')

    print('\t\t\\midrule')
    print(f'\t\t\\textbf{{Total}} & \\textbf{{{sum(pair_counts.values())}}} \\\\')
    print('\t\t\\bottomrule')
    print('\t\end{tabular}')

if __name__ == '__main__':
    selected_models = helper.get_selected_repos()

    mismatched_sections = get_corrections(selected_models['model_id'].tolist())
    print_section_subsection_table(mismatched_sections)
    print_subsection_table(mismatched_sections)