"""webapps URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from koppers import views as kopper_views

urlpatterns = [
    path("admin/", admin.site.urls),
    path('kopper/', kopper_views.index_action),
    path('kopper/dashboard-admin', kopper_views.dashboard_admin_action),
    path('kopper/new-calculation', kopper_views.new_calculation_action),
    path('kopper/add-new-ties', kopper_views.add_new_ties_action),
    path('kopper/calculation-result', kopper_views.calculation_result_action, name='calculation-result'),
]
