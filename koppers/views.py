from django.shortcuts import render


# Create your views here.

def present_action(request):
    if request.method == "GET":
        context = {"status": "Welcome to Kopper Capstone Project."}
        return render(request, "index.html", context)

    return
