from django import forms

class CSVUploadForm(forms.Form):
    csv_file = forms.FileField(label='CSV File', required=True)
    text_box_1 = forms.IntegerField(required=True)
    text_box_2 = forms.IntegerField(required=True)
    dropdown_box_1 = forms.ChoiceField(choices=[(i, i) for i in range(1, 7)], initial=5)
    dropdown_box_2 = forms.ChoiceField(choices=[(i, i) for i in range(1, 7)], initial=5)
    text_box_5 = forms.CharField(required=False, label='Text Box 5', max_length=100)
    text_box_6 = forms.DecimalField(required=True) # weight factor
