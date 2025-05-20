import pandas as pd
import numpy as np

def map_age(age):
    if age <= 30:
        return 5
    elif age <= 40:
        return 4
    elif age <= 50:
        return 3
    else:
        return 2

def map_household_size(size):
    if size <= 2:
        return 1
    elif size <= 5:
        return 3
    else:
        return 2

def map_household_composition(ratio):
    if ratio < 70:
        return 5
    elif ratio < 85:
        return 3
    else:
        return 1

def map_monthly_household_income(income):
    if income < 12000:
        return 1
    elif income < 20000:
        return 3
    else:
        return 5

def map_dependents_education(count):
    # If you want a fallback for unknown counts, adjust accordingly
    lookup = {
        0: 1,
        1: 2,
        2: 3,
        3: 4,
        4: 5,
        5: 6
    }
    return lookup.get(count, 0)

###############################################################################
# Asset Mapping
###############################################################################
def map_land_area_ha(area):
    if area <= 1.2:
        return 2
    elif area <= 2.0:
        return 3
    else:
        return 5

def map_number_of_cattle(n):
    if n <= 2:
        return 1
    elif n <= 5:
        return 3
    else:
        return 5

def map_number_of_sheep(n):
    if n <= 2:
        return 1
    elif n <= 5:
        return 3
    else:
        return 5

def map_number_of_goats(n):
    if n <= 1:
        return 1
    elif n <= 3:
        return 3
    else:
        return 5

def map_number_of_poultry(n):
    if n <= 2:
        return 1
    elif n <= 6:
        return 3
    else:
        return 5

def map_monthly_ag_expenditure(perc):
    if perc < 15:
        return 1
    elif perc < 25:
        return 3
    else:
        return 5
    
def map_proximity_to_markets(d):
    if d <= 5:
        return 5
    elif d <= 15:
        return 3
    else:
        return 1