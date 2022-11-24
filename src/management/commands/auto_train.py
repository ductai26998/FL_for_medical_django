from django.core.management.base import BaseCommand
from ...center_training.utils import train_center

class Command(BaseCommand):
    def handle(self, *args, **kwargs):
        train_center()