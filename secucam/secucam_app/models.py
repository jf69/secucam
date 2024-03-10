from django.db import models

# Database table that stores the footage names
class Footage(models.Model):
	footage_name = models.CharField(max_length=200, null=False, blank=False, db_index=True)
	# When a row from this table is returned, return the string value of the footage_name instead of the PK.
	def __str__(self):
		return self.footage_name