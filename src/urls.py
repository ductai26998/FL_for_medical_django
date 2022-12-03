"""src URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
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
from django.conf.urls import url
from django.contrib import admin
from django.urls import path

# from .center.views import (CenterGetClientParams, CenterReceivesParams,
#                            CenterSendsParams)
# from .client.views import ClientReceivesParams, ClientSendsParams
from .device.views import (CenterGetClientParams, CenterReceivesParams,
                           CenterSendsParams, ClientReceivesParams,
                           ClientSendsParams)

urlpatterns = [
    path('admin/', admin.site.urls),
    url("center/params/sends", CenterSendsParams.as_view(),
        name="center_sends_params"),
    url("center/params/receives", CenterReceivesParams.as_view(),
        name="center_receives_params"),
    url("center/params", CenterGetClientParams.as_view(), name="center_get_params"),
    # FIXME: move client urls to client device
    url("client/params/sends", ClientSendsParams.as_view(),
        name="client_sends_params"),
    url("client/params/receives", ClientReceivesParams.as_view(),
        name="client_receives_params"),
]
