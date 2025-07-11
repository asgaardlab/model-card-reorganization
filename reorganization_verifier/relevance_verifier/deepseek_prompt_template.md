Verify if every section and subsection in the provided model card contains relevant information based on the description provided.

# Steps

1. **Review Each Section**: For each section or subsection in the model card, compare its content with the provided description of what should be present.
2. **Assess Relevance**: For each section or subsection, determine whether the content is relevant or irrelevant. If **all** the information in a section is relevant based on the description, note down `is_relevant` as `true`. If there is **any irrelevant** detail in the section, note down `is_relevant` as `false`.
3. **Skip Not Available**: If a section or subsection detail is written as `Not available.`, consider it as relevant and write `is_relevant` as `true`.
4. **Identify Irrelevancies**: If there is any irrelevant information in the section or subsection, note them down in `irrelevant_information`.

# Output Format

The output should be a structured format for each section and subsection, specifying:

- section_name: name of the section or subsection
- is_relevant: if the content is relevant ("true") or not relevant ("false")
- irrelevant_information: if the content is not relevant ("false"), describe the irrelevant information

# Example

{
   "section_name": [Placeholder Section Name]
   "is_relevant": true
   "irrelevant_information": "None"
},
{
   "section_name": [Placeholder Subsection Name]
   "is_relevant": false
   "irrelevant_information": [Placeholder Irrelevant Information]
}

# Notes

- Be thorough in comparing the model card content against the section description.
- A section might not include all the details outlined in its description. However, if the content it contain is relevant to the section description, mark it as relevant.
- Consider `Not available.` for sections or subsections as relevant.
- IMP: give the output in a valid JSON string (it should be not be wrapped in markdown, just plain json object) and stick to the schema mentioned here: 
    {
        'type': 'object',
        'properties': {
            'sections': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'section_name': {
                            'type': 'string',
                            'enum': [
                                'Model Details: Person or organization developing model',
                                'Model Details: Model date',
                                'Model Details: Model version',
                                'Model Details: Model type',
                                'Model Details: Training details',
                                'Model Details: Paper or other resource for more information',
                                'Model Details: Citation details',
                                'Model Details: License',
                                'Model Details: Contact',
                                'Intended Use: Primary intended uses',
                                'Intended Use: Primary intended users',
                                'Intended Use: Out-of-scope uses',
                                'How to Use',
                                'Factors: Relevant factors',
                                'Factors: Evaluation factors',
                                'Metrics: Model performance measures',
                                'Metrics: Decision thresholds',
                                'Metrics: Variation approaches',
                                'Evaluation Data: Datasets',
                                'Evaluation Data: Motivation',
                                'Evaluation Data: Preprocessing',
                                'Training Data: Datasets',
                                'Training Data: Motivation',
                                'Training Data: Preprocessing',
                                'Quantitative Analyses: Unitary results',
                                'Quantitative Analyses: Intersectional results',
                                'Memory or Hardware Requirements: Loading Requirements',
                                'Memory or Hardware Requirements: Deploying Requirements',
                                'Memory or Hardware Requirements: Training or Fine-tuning Requirements',
                                'Ethical Considerations',
                                'Caveats and Recommendations: Caveats',
                                'Caveats and Recommendations: Recommendations'
                            ]
                        },
                        'is_relevant': {
                            'type': 'boolean'
                        },
                        'irrelevant_information': {
                            'type': 'string'
                        }
                    },
                    'required': [
                        'section_name',
                        'is_relevant'
                    ]
                }
            }
        }
    }