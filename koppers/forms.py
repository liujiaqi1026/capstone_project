from django import forms
import csv
class CSVUploadForm(forms.Form):
    csv_file = forms.FileField(label='CSV File', required=True)
    text_box_1 = forms.IntegerField(required=True)
    text_box_2 = forms.IntegerField(required=True)
    dropdown_box_1 = forms.ChoiceField(choices=[(i, i) for i in range(1, 7)], initial=5)
    dropdown_box_2 = forms.ChoiceField(choices=[(i, i) for i in range(1, 7)], initial=5)
    text_box_5 = forms.CharField(required=False, label='Text Box 5', max_length=100)
    text_box_6 = forms.DecimalField(required=True) # weight factor
    radio = forms.CharField(required=True, max_length=100)

    # Add a clean method for csv_file field to validate file format
    def clean_csv_file(self):
        file = self.cleaned_data.get('csv_file')
        if file:
            if not file.name.endswith('.csv'):
                raise forms.ValidationError('File format must be CSV.')
            
            # Open file in text mode
            data = file.read().decode('utf-8').splitlines()
            reader = csv.reader(data)
            headers = next(reader)  # Skip header row
            rows = list(reader)
            
            # Check if all non-zero quantities have the same height and width
            non_zero_rows = [row for row in rows if float(row[3]) != 0]
            if non_zero_rows:
                height = float(non_zero_rows[0][0])
                width = float(non_zero_rows[0][1])
                for row in non_zero_rows:
                    if float(row[0]) != height or float(row[1]) != width:
                        raise forms.ValidationError('All input ties must have the same height and width.')
            
            # Reset file pointer to beginning
            file.seek(0)
        return file
