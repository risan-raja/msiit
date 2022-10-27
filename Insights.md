- Customer Type with children doesnt coincide with having children
- Divorced has the highest percent with kids
- Changed Destination to have ordinal Meaning with Travel Time aggregate
- Combining directly opposite features to give nominal features with varied target expectations.
- reduced more 
```python
    prefs = [
        "binary__prefer_home_food",
        "binary__is_foodie",
        "binary__prefer_western_over_chinese",
        "binary__cooks_regularly",
    ]
    
    "biz_type = (nominal__restaurant_type * 2) + (binary__offer_expiration)"
    spend_id =  [
            "ordinal__restaur_spend_greater_than20",
            "ordinal__restaur_spend_less_than20",
        ]
    direction_f = [
        "binary__restuarant_opposite_direction_house",
        "binary__restuarant_same_direction_house",
    ]
    extra_effort = [
        "binary__travelled_more_than_15mins_for_offer",
        "binary__travelled_more_than_25mins_for_offer",
    ]
     circumstance = ["nominal__customer_type", "nominal__extra_travel", "nominal__direction"]
```
```text
cutomer_type Legend

0: Individual
1: With Family
2: With Kids
3: With Colleagues

marital_status Legend

0: Married Partner
1: Single
2: Divorced
3: UnMarried Partner
4: Widowed

Restaurant Type Legend

"Cold drinks"             :2
"2 star restaurant "      :4
"Take-away restaurant"    :1
"4 star restaurant   "    :0
"Restaurant with pub  "   :3



{'Unemployed': 0,
 'Student': 3,
 'Computer & Mathematical': 5,
 'Sales & Related': 2,
 'Education&Training&Library': 13,
 'Management': 7,
 'Arts Design Entertainment Sports & Media': 1,
 'Office & Administrative Support': 6,
 'Business & Financial': 4,
 'Retired': 17,
 'Food Preparation & Serving Related': 23,
 'Healthcare Practitioners & Technical': 15,
 'Transportation & Material Moving': 16,
 'Community & Social Services': 12,
 'Healthcare Support': 8,
 'Legal': 11,
 'Protective Service': 21,
 'Life Physical Social Science': 9,
 'Personal Care & Service': 22,
 'Architecture & Engineering': 18,
 'Construction & Extraction': 14,
 'Installation Maintenance & Repair': 10,
 'Production Occupations': 19,
 'Building & Grounds Cleaning & Maintenance': 24,
 'Farming Fishing & Forestry': 20}

```
```
Potential Grouping:
- Pref Profile
- Biz Info
```