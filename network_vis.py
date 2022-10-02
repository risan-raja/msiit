from pyvis import network as net
import networkx as nx

relations = {
    "binary__offer_expiration": {"nominal__biz_type": {"weight": 1.0}},
    "ordinal__income_range": {
        "nominal__marital_status": {"weight": 0.3229227552740118},
        "nominal__job_industry": {"weight": 0.4544411182784494},
        "nominal__spend_id": {"weight": 0.38496093241066615},
    },
    "ordinal__no_visited_cold_drinks": {
        "ordinal__restaur_spend_less_than20": {"weight": 0.4307193245644276},
        "ordinal__no_visited_bars": {"weight": 0.3580911970493063},
        "ordinal__no_take_aways": {"weight": 0.32854919078061934},
        "nominal__job_industry": {"weight": 0.35230763109679564},
        "ordinal__restaur_spend_greater_than20": {"weight": 0.33056023085873376},
        "nominal__spend_id": {"weight": 0.46743900523069826},
    },
    "binary__travelled_more_than_15mins_for_offer": {
        "binary__travelled_more_than_25mins_for_offer": {"weight": 0.39659198425640746},
        "binary__restuarant_same_direction_house": {"weight": 0.36082474237014767},
        "binary__restuarant_opposite_direction_house": {"weight": 0.37431108765131293},
        "nominal__direction": {"weight": 0.3802218972541071},
        "nominal__extra_travel": {"weight": 1.0},
    },
    "ordinal__restaur_spend_less_than20": {
        "ordinal__no_visited_cold_drinks": {"weight": 0.4307193245644276},
        "ordinal__no_take_aways": {"weight": 0.4597798493308146},
        "nominal__job_industry": {"weight": 0.34158198932471573},
        "ordinal__restaur_spend_greater_than20": {"weight": 0.5452883194332624},
        "nominal__spend_id": {"weight": 1.0},
    },
    "nominal__marital_status": {
        "ordinal__income_range": {"weight": 0.3229227552740118},
        "ordinal__age": {"weight": 0.3452504526896679},
        "ordinal__no_visited_bars": {"weight": 0.3132691430226643},
        "nominal__job_industry": {"weight": 0.49449785701688526},
        "binary__has_children": {"weight": 0.36953590784518564},
    },
    "nominal__restaurant_type": {"nominal__biz_type": {"weight": 1.0}},
    "ordinal__age": {
        "nominal__marital_status": {"weight": 0.3452504526896679},
        "nominal__job_industry": {"weight": 0.5790477295300044},
        "binary__has_children": {"weight": 0.520951715237399},
        "nominal__spend_id": {"weight": 0.36622371949806276},
    },
    "binary__prefer_western_over_chinese": {"nominal__pref_profile": {"weight": 1.0}},
    "binary__travelled_more_than_25mins_for_offer": {
        "binary__travelled_more_than_15mins_for_offer": {"weight": 0.39659198425640746},
        "nominal__extra_travel": {"weight": 1.0},
    },
    "ordinal__no_visited_bars": {
        "ordinal__no_visited_cold_drinks": {"weight": 0.3580911970493063},
        "nominal__marital_status": {"weight": 0.3132691430226643},
        "ordinal__no_take_aways": {"weight": 0.3169497483921236},
        "nominal__job_industry": {"weight": 0.38937305246747433},
        "ordinal__restaur_spend_greater_than20": {"weight": 0.4367909681084544},
        "nominal__spend_id": {"weight": 0.4694534754416848},
    },
    "binary__restuarant_same_direction_house": {
        "binary__travelled_more_than_15mins_for_offer": {"weight": 0.36082474237014767},
        "nominal__customer_type": {"weight": 0.3272584646385474},
        "binary__restuarant_opposite_direction_house": {"weight": 0.9550712371834301},
        "nominal__direction": {"weight": 1.0},
        "nominal__extra_travel": {"weight": 0.3813737216747287},
    },
    "binary__cooks_regularly": {"nominal__pref_profile": {"weight": 1.0}},
    "nominal__customer_type": {
        "binary__restuarant_same_direction_house": {"weight": 0.3272584646385474},
        "binary__restuarant_opposite_direction_house": {"weight": 0.3229526808041737},
        "nominal__direction": {"weight": 0.3254355914152093},
        "nominal__extra_travel": {"weight": 0.3168853370794687},
    },
    "ordinal__qualif": {
        "nominal__job_industry": {"weight": 0.4401927965907077},
        "nominal__spend_id": {"weight": 0.36599216970352944},
    },
    "binary__is_foodie": {"nominal__pref_profile": {"weight": 1.0}},
    "ordinal__no_take_aways": {
        "ordinal__no_visited_cold_drinks": {"weight": 0.32854919078061934},
        "ordinal__restaur_spend_less_than20": {"weight": 0.4597798493308146},
        "ordinal__no_visited_bars": {"weight": 0.3169497483921236},
        "nominal__job_industry": {"weight": 0.40264912838717787},
        "nominal__spend_id": {"weight": 0.48905271250323973},
    },
    "nominal__job_industry": {
        "ordinal__income_range": {"weight": 0.4544411182784494},
        "ordinal__no_visited_cold_drinks": {"weight": 0.35230763109679564},
        "ordinal__restaur_spend_less_than20": {"weight": 0.34158198932471573},
        "nominal__marital_status": {"weight": 0.49449785701688526},
        "ordinal__age": {"weight": 0.5790477295300044},
        "ordinal__no_visited_bars": {"weight": 0.38937305246747433},
        "ordinal__qualif": {"weight": 0.4401927965907077},
        "ordinal__no_take_aways": {"weight": 0.40264912838717787},
        "ordinal__restaur_spend_greater_than20": {"weight": 0.35535384790412794},
        "nominal__spend_id": {"weight": 0.5533750484766163},
    },
    "binary__restuarant_opposite_direction_house": {
        "binary__travelled_more_than_15mins_for_offer": {"weight": 0.37431108765131293},
        "binary__restuarant_same_direction_house": {"weight": 0.9550712371834301},
        "nominal__customer_type": {"weight": 0.3229526808041737},
        "nominal__direction": {"weight": 1.0},
        "nominal__extra_travel": {"weight": 0.3917209318653619},
    },
    "binary__has_children": {
        "nominal__marital_status": {"weight": 0.36953590784518564},
        "ordinal__age": {"weight": 0.520951715237399},
    },
    "interval__temperature": {"interval__season": {"weight": 0.4842975526042193}},
    "ordinal__restaur_spend_greater_than20": {
        "ordinal__no_visited_cold_drinks": {"weight": 0.33056023085873376},
        "ordinal__restaur_spend_less_than20": {"weight": 0.5452883194332624},
        "ordinal__no_visited_bars": {"weight": 0.4367909681084544},
        "nominal__job_industry": {"weight": 0.35535384790412794},
        "nominal__spend_id": {"weight": 1.0},
    },
    "interval__travel_time": {"ordinal__dest_distance": {"weight": 0.3913082317384624}},
    "interval__season": {"interval__temperature": {"weight": 0.4842975526042193}},
    "ordinal__dest_distance": {"interval__travel_time": {"weight": 0.3913082317384624}},
    "binary__prefer_home_food": {"nominal__pref_profile": {"weight": 1.0}},
    "nominal__pref_profile": {
        "binary__prefer_western_over_chinese": {"weight": 1.0},
        "binary__cooks_regularly": {"weight": 1.0},
        "binary__is_foodie": {"weight": 1.0},
        "binary__prefer_home_food": {"weight": 1.0},
    },
    "nominal__biz_type": {
        "binary__offer_expiration": {"weight": 1.0},
        "nominal__restaurant_type": {"weight": 1.0},
    },
    "nominal__spend_id": {
        "ordinal__income_range": {"weight": 0.38496093241066615},
        "ordinal__no_visited_cold_drinks": {"weight": 0.46743900523069826},
        "ordinal__restaur_spend_less_than20": {"weight": 1.0},
        "ordinal__age": {"weight": 0.36622371949806276},
        "ordinal__no_visited_bars": {"weight": 0.4694534754416848},
        "ordinal__qualif": {"weight": 0.36599216970352944},
        "ordinal__no_take_aways": {"weight": 0.48905271250323973},
        "nominal__job_industry": {"weight": 0.5533750484766163},
        "ordinal__restaur_spend_greater_than20": {"weight": 1.0},
    },
    "nominal__direction": {
        "binary__travelled_more_than_15mins_for_offer": {"weight": 0.3802218972541071},
        "binary__restuarant_same_direction_house": {"weight": 1.0},
        "nominal__customer_type": {"weight": 0.3254355914152093},
        "binary__restuarant_opposite_direction_house": {"weight": 1.0},
        "nominal__extra_travel": {"weight": 0.40066263046193207},
    },
    "nominal__extra_travel": {
        "binary__travelled_more_than_15mins_for_offer": {"weight": 1.0},
        "binary__travelled_more_than_25mins_for_offer": {"weight": 1.0},
        "binary__restuarant_same_direction_house": {"weight": 0.3813737216747287},
        "nominal__customer_type": {"weight": 0.3168853370794687},
        "binary__restuarant_opposite_direction_house": {"weight": 0.3917209318653619},
        "nominal__direction": {"weight": 0.40066263046193207},
    },
}

G = nx.Graph(relations)

nx.draw(G, with_labels=True)

options = """
const options = {
  "nodes": {
    "borderWidth": null,
    "borderWidthSelected": null,
    "opacity": null,
    "size": null
  },
  "edges": {
    "arrows": {
      "to": {
        "enabled": true,
        "scaleFactor": 1.35
      },
      "from": {
        "enabled": true
      }
    },
    "color": {
      "inherit": true
    },
    "scaling": {
      "min": 6
    },
    "selfReferenceSize": null,
    "selfReference": {
      "angle": 0.7853981633974483
    },
    "smooth": false
  },
  "layout": {
    "hierarchical": {
      "enabled": true,
      "levelSeparation": 245,
      "nodeSpacing": 185,
      "treeSpacing": 305,
      "blockShifting": false,
      "edgeMinimization": false,
      "parentCentralization": false,
      "sortMethod": "directed",
      "shakeTowards": "roots"
    }
  },
  "interaction": {
    "hover": true,
    "keyboard": {
      "enabled": true
    },
    "multiselect": true,
    "navigationButtons": true,
    "tooltipDelay": 75
  },
  "manipulation": {
    "enabled": true,
    "initiallyActive": true
  },
  "physics": {
    "hierarchicalRepulsion": {
      "centralGravity": 0,
      "springLength": 285,
      "springConstant": 0.01,
      "nodeDistance": 140,
      "damping": 1,
      "avoidOverlap": 0.92
    },
    "maxVelocity": 0.02,
    "minVelocity": 0.01,
    "solver": "hierarchicalRepulsion"
  }
}
"""
g = net.Network(notebook=True)
g.set_options(options)
# g.show_buttons()
g.from_nx(G)
g.write_html("exp.html")
# g.show("ex.html")
