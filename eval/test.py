import pandas as pd

merged = pd.read_excel('advanced-prompting/merged_answers.xlsx')
merged['sample_id'] = merged['PMID'].astype(str)+'-'+merged['QID'].astype(str)

detail = pd.read_csv('eval/detailed_evaluation.csv')
human = detail[detail['Scenario']=='Human Answer'].copy()
human['sample_id'] = human['PMID'].astype(str)+'-'+human['QID'].astype(str)

legacy = merged.set_index('sample_id')['GPT-4o base correct']
human['legacy_correct'] = human['sample_id'].map(legacy)

mismatches = human[human['legacy_correct'] != human['GPT-4o base Correct']]
output = mismatches[['PMID','QID','Question','Human Answer',
                    'GPT-4o base Answer','legacy_correct','GPT-4o base Correct']].to_string(index=False)
print(output)
with open('mismatch_report.txt', 'w', encoding='utf-8') as outfile:
    outfile.write(output + '\n')
