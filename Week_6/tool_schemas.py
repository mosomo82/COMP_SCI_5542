tool_schemas = [
    {
        "type": "function",
        "function": {
            "name": "query_snowflake",
            "description": "Executes a read-only SQL query against the Snowflake database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql_query": {
                        "type": "string",
                        "description": "The SQL query string to execute. Example: 'SELECT * FROM CS5542_WEEK5.PUBLIC.TRUCKS LIMIT 5;'"
                    }
                },
                "required": ["sql_query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_monthly_revenue",
            "description": "Retrieves aggregated monthly revenue trends within a specified date range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_month": {
                        "type": "string",
                        "description": "Start month in 'YYYY-MM-DD' format (e.g., '2023-01-01')."
                    },
                    "end_month": {
                        "type": "string",
                        "description": "End month in 'YYYY-MM-DD' format (e.g., '2025-12-31')."
                    }
                },
                "required": ["start_month", "end_month"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_fleet_performance",
            "description": "Retrieves truck performance metrics based on specified filters like minimum trips, fuel types, and limits the output to top N performing trucks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "min_trips": {
                        "type": "integer",
                        "description": "Minimum number of trips completed by a truck to be included in the aggregation. Defaults to 5."
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Maximum number of top-performing trucks to return. Defaults to 30."
                    },
                    "fuel_types": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "List of fuel types to include. Expected values: 'Diesel', 'CNG', 'Electric'. Defaults to all if not provided."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_pipeline_logs",
            "description": "Reads the automated ingestion pipeline logs to return system health and latency data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of recent log entries to return. Defaults to 100."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_safety_metrics",
            "description": "Retrieves safety incident metrics for top drivers based on filters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "incident_types": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "List of incident types to include (e.g. 'Collision', 'Near Miss', 'Moving Violation'). Defaults to all if not provided."
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Start date in 'YYYY-MM-DD' format. Defaults to '2022-01-01'."
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date in 'YYYY-MM-DD' format. Defaults to '2025-12-31'."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of top drivers by incident count to return. Defaults to 15."
                    }
                }
            }
        }
    }
    ,
    {
        "type": "function",
        "function": {
            "name": "get_route_profitability",
            "description": "Retrieves route profitability metrics from V_ROUTE_SCORECARD, including revenue, fuel cost, gross profit, margin percentage, and average MPG per route.",
            "parameters": {
                "type": "object",
                "properties": {
                    "min_loads": {
                        "type": "integer",
                        "description": "Minimum number of completed loads for a route to be included. Defaults to 3."
                    },
                    "min_margin_pct": {
                        "type": "number",
                        "description": "Minimum gross margin percentage to filter routes. Defaults to 0.0."
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Maximum number of routes to return, sorted by gross profit descending. Defaults to 20."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_delivery_performance",
            "description": "Retrieves delivery event performance aggregated by city and state, including on-time rates and average detention times from the DELIVERY_EVENTS table.",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_type": {
                        "type": "string",
                        "description": "Type of delivery event to filter on. Expected values: 'Delivery' or 'Pickup'. Defaults to 'Delivery'."
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Start date in 'YYYY-MM-DD' format. Defaults to '2022-01-01'."
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date in 'YYYY-MM-DD' format. Defaults to '2025-12-31'."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of city/state rows to return, ordered by total events. Defaults to 20."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_maintenance_health",
            "description": "Retrieves truck maintenance health metrics including costs, downtime, and event counts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "maintenance_type": {
                        "type": "string",
                        "description": "'Scheduled', 'Unscheduled', or 'Inspection'. None = all. Defaults to all if not provided."
                    },
                    "start_date": {
                        "type": "date",
                        "description": "Start date in 'YYYY-MM-DD' format (e.g., '2023-01-01')."
                    },
                    "end_date": {
                        "type": "date",
                        "description": "End date in 'YYYY-MM-DD' format (e.g., '2025-12-31')."
                    },
                    "top_n":{
                        "type": "integer",
                        "description": "Maximum number of trucks to return, sorted by total cost. Defaults to 20."
                    } 
                },
                "required": ["start_month", "end_month"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_fuel_spend_analysis",
            "description": "Retrieves fuel spend analysis aggregated by state and city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "states": {
                        "type": "list",
                        "description": "State abbreviations to filter (e.g. ['TX','CA']). None = all."
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Max locations to return. Defaults to 15."
                    }
                },
                "required": ["states", "top_n"]
            }
        }
    }    
]
