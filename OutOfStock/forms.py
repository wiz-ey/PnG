from django import forms

class TideForm(forms.Form):

    current_inventory = forms.IntegerField()

class FaceNetForm(forms.Form):

    image = forms.CharField(max_length=255)
