import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Any, Optional
import re
import traceback
import json
import logging
import copy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NLPVisualizer:
    def __init__(self, chat_module):
        self.chat_module = chat_module
        self._column_cache = {}  # Add caching for performance
        self.chart_types = {
            'line': self._create_line_chart,
            'bar': self._create_bar_chart,
            'scatter': self._create_scatter_chart,
            'pie': self._create_pie_chart,
            'histogram': self._create_histogram,
            'box': self._create_box_plot,
            'combo': self._create_combo_chart,
            'yoy': self._create_yoy_chart
        }
        
        self.chart_descriptions = {
            'line': "Line charts show trends over time or continuous variables",
            'bar': "Bar charts compare quantities across categories",
            'scatter': "Scatter plots show relationships between two variables",
            'pie': "Pie charts show proportions of a whole",
            'histogram': "Histograms show distributions of a single variable",
            'box': "Box plots show statistical distributions with outliers",
            'combo': "Combo charts combine bar and line charts to display related metrics"
        }
    
    def _sanitize_for_regex(self, text: str) -> str:
        """Sanitize text for safe use in regex patterns"""
        return re.escape(text)
    
    def _get_date_columns(self, df: pd.DataFrame) -> List[str]:
        """Centralized method to detect date columns"""
        date_columns = []
        
        # First, check for already datetime columns
        date_columns.extend(df.select_dtypes(include=['datetime64']).columns.tolist())
        
        # Then try to convert potential date columns
        for col in df.columns:
            if col not in date_columns and df[col].dtype == 'object':
                if any(term in col.lower() for term in ['date', 'time', 'day', 'month', 'year']):
                    try:
                        # Test conversion on a small sample
                        sample = df[col].dropna().head(10)
                        if len(sample) > 0:
                            converted = pd.to_datetime(sample, errors='coerce')
                            if not converted.isna().all():
                                date_columns.append(col)
                    except:
                        pass
        
        return date_columns
    
    def _get_column_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Cache column type detection"""
        df_id = id(df)
        if df_id not in self._column_cache:
            self._column_cache[df_id] = {
                'numeric': df.select_dtypes(include=[np.number]).columns.tolist(),
                'categorical': df.select_dtypes(include=['object', 'category']).columns.tolist(),
                'datetime': self._get_date_columns(df)
            }
        return self._column_cache[df_id]
    
    def _get_reference_year(self, df: pd.DataFrame) -> int:
        """Get the most appropriate reference year from data"""
        date_columns = self._get_date_columns(df)
        
        if date_columns:
            try:
                # Get the most recent year from the first date column
                latest_date = df[date_columns[0]].max()
                if pd.notna(latest_date):
                    return latest_date.year
            except:
                pass
        
        # Fallback to current year
        return pd.Timestamp.now().year
    
    def _validate_config_and_data(self, df: pd.DataFrame, config: Dict[str, Any], required_keys: List[str]) -> Optional[str]:
        """Validate configuration and data compatibility"""
        # Check required config keys
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            return f"Missing required config keys: {missing_keys}"
        
        # Check if dataframe is empty
        if df.empty:
            return "DataFrame is empty"
        
        # Validate columns exist in dataframe
        for key in ['x_column', 'y_column']:
            if key in config:
                columns = config[key]
                if isinstance(columns, str):
                    columns = [columns]
                elif isinstance(columns, list):
                    pass
                else:
                    continue
                
                missing_cols = [col for col in columns if col and col not in df.columns]
                if missing_cols:
                    return f"Columns not found in data: {missing_cols}"
        
        return None
    
    def _create_chart_with_error_handling(self, chart_func, df: pd.DataFrame, config: Dict[str, Any], chart_type: str) -> go.Figure:
        """Wrapper for chart creation with comprehensive error handling"""
        try:
            return chart_func(df, config)
        except Exception as e:
            error_context = {
                'chart_type': chart_type,
                'df_shape': df.shape,
                'df_columns': list(df.columns),
                'config_keys': list(config.keys()),
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            logger.error(f"Chart creation failed: {json.dumps(error_context, indent=2)}")
            
            return go.Figure().add_annotation(
                text=f"Chart creation failed: {str(e)}<br>Please check your data and configuration",
                showarrow=False,
                font=dict(size=14, color="red"),
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper"
            )
    
    def convert_timestamps_to_strings(self, config):
        """Convert any pandas Timestamp objects to string format safely"""
        if isinstance(config, dict):
            result = {}
            for key, value in config.items():
                if isinstance(value, pd.Timestamp):
                    result[key] = value.strftime('%Y-%m-%d')
                elif isinstance(value, (dict, list)):
                    result[key] = self.convert_timestamps_to_strings(copy.deepcopy(value))
                else:
                    result[key] = value
            return result
        elif isinstance(config, list):
            result = []
            for item in config:
                if isinstance(item, pd.Timestamp):
                    result.append(item.strftime('%Y-%m-%d'))
                elif isinstance(item, (dict, list)):
                    result.append(self.convert_timestamps_to_strings(copy.deepcopy(item)))
                else:
                    result.append(item)
            return result
        return config
    
    def _convert_figure_to_chart_js_format(self, fig):
        """Convert Plotly figure to Chart.js compatible format with proper error handling"""
        try:
            # FIXED: Add proper bounds checking
            if not fig or not hasattr(fig, 'data') or not fig.data or len(fig.data) == 0:
                logger.warning("Figure has no data to convert")
                return {
                    'data': {'x': [], 'y': [], 'labels': [], 'values': []},
                    'additionalDatasets': [],
                    'type': 'standard'
                }
            
            # Check if this is a pie chart or combo chart
            is_pie_chart = False
            is_combo_chart = False
            
            if hasattr(fig.data[0], 'type'):
                if fig.data[0].type == 'pie':
                    is_pie_chart = True
                elif len(fig.data) > 1:
                    # Check for mix of bar and scatter traces
                    trace_types = [trace.type for trace in fig.data if hasattr(trace, 'type')]
                    if 'bar' in trace_types and 'scatter' in trace_types:
                        is_combo_chart = True
                        
            # Get data directly from the figure object
            plotly_data = []
            for trace in fig.data:
                trace_dict = {}
                
                # Handle pie charts differently
                if is_pie_chart:
                    # Extract labels
                    if hasattr(trace, 'labels') and trace.labels is not None:
                        if isinstance(trace.labels, (list, np.ndarray)):
                            trace_dict['labels'] = list(trace.labels)
                    
                    # Extract values
                    if hasattr(trace, 'values') and trace.values is not None:
                        if isinstance(trace.values, (list, np.ndarray)):
                            trace_dict['values'] = list(trace.values)
                    
                    # Add name if available
                    if hasattr(trace, 'name') and trace.name:
                        trace_dict['name'] = trace.name
                        
                    plotly_data.append(trace_dict)
                else:
                    # Standard handling for other charts
                    # Add trace type for combo chart identification
                    if hasattr(trace, 'type'):
                        trace_dict['type'] = trace.type
                    
                    # Extract x data
                    if hasattr(trace, 'x') and trace.x is not None:
                        if isinstance(trace.x, (list, np.ndarray)):
                            trace_dict['x'] = list(trace.x)
                            # Convert datetime to strings if needed
                            if trace_dict['x'] and isinstance(trace_dict['x'][0], (np.datetime64, pd.Timestamp)):
                                trace_dict['x'] = [pd.Timestamp(x).isoformat() for x in trace_dict['x']]
                    
                    # Extract y data
                    if hasattr(trace, 'y') and trace.y is not None:
                        if isinstance(trace.y, (list, np.ndarray)):
                            trace_dict['y'] = list(trace.y)
                    
                    # Add other relevant trace properties
                    if hasattr(trace, 'name') and trace.name:
                        trace_dict['name'] = trace.name
                    
                    # For combo charts - capture yaxis info
                    if hasattr(trace, 'yaxis'):
                        trace_dict['yaxis'] = trace.yaxis
                    
                    # Get color information if available
                    if hasattr(trace, 'marker') and hasattr(trace.marker, 'color'):
                        trace_dict['color'] = trace.marker.color
                        
                    plotly_data.append(trace_dict)
            
            # Process for Chart.js format
            if not plotly_data:
                return {
                    'data': {'x': [], 'y': [], 'labels': [], 'values': []},
                    'additionalDatasets': [],
                    'type': 'standard'
                }
                
            if is_pie_chart:
                chart_data = {
                    'labels': plotly_data[0].get('labels', []),
                    'values': plotly_data[0].get('values', []),
                    'x': plotly_data[0].get('labels', []),
                    'y': plotly_data[0].get('values', [])
                }
                
                return {
                    'data': chart_data,
                    'additionalDatasets': [],
                    'type': 'pie'
                }
            elif is_combo_chart:
                labels = []
                for trace in plotly_data:
                    if 'x' in trace:
                        labels = trace['x']
                        break
                bar_datasets = []
                line_datasets = []
                
                for trace in plotly_data:
                    dataset = {
                        'label': trace.get('name', ''),
                        'data': trace.get('y', []),
                        'yAxisID': 'y2' if trace.get('yaxis') == 'y2' else 'y'
                    }
                    
                    # Add color if available
                    if 'color' in trace:
                        dataset['backgroundColor'] = trace['color']
                        dataset['borderColor'] = trace['color']
                    
                    if trace.get('type') == 'bar':
                        dataset['type'] = 'bar'
                        bar_datasets.append(dataset)
                    elif trace.get('type') == 'scatter':
                        dataset['type'] = 'line'
                        line_datasets.append(dataset)
                
                return {
                    'data': {
                        'labels': labels,
                        'datasets': bar_datasets + line_datasets
                    },
                    'additionalDatasets': [],
                    'type': 'combo',
                    'useSecondaryYAxis': any(trace.get('yaxis') == 'y2' for trace in plotly_data)
                }
            else:
                # For standard charts
                chart_data = {
                    'x': plotly_data[0].get('x', []),
                    'y': plotly_data[0].get('y', []),
                    'labels': plotly_data[0].get('x', []),
                    'values': plotly_data[0].get('y', [])
                }
                
                # Additional datasets from other traces
                additional_datasets = []
                for i in range(1, len(plotly_data)):
                    additional_datasets.append({
                        'name': plotly_data[i].get('name', f'Series {i+1}'),
                        'x': plotly_data[i].get('x', chart_data['x']),
                        'y': plotly_data[i].get('y', []),
                        'color': plotly_data[i].get('color')
                    })
                
                return {
                    'data': chart_data,
                    'additionalDatasets': additional_datasets,
                    'type': 'standard'
                }
                
        except Exception as e:
            logger.error(f"Error converting figure to Chart.js format: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Return minimal valid data structure to prevent frontend errors
            return {
                'data': {
                    'x': [],
                    'y': [],
                    'labels': [],
                    'values': []
                },
                'additionalDatasets': []
            }

    def process_visualization_prompt(self, df: pd.DataFrame, prompt: str) -> Dict[str, Any]:
        """Process natural language prompt and return visualization configuration"""
        try:
            # Validate input
            if df.empty:
                return {
                    'success': False,
                    'error': 'DataFrame is empty'
                }
            
            if not prompt or not prompt.strip():
                return {
                    'success': False,
                    'error': 'Prompt cannot be empty'
                }
            
            # Step 1: Determine the appropriate chart type and data columns
            chart_config = self._get_chart_configuration(df, prompt)
            
            if not chart_config['success']:
                return chart_config
            
            # Step 2: Generate the visualization
            visual_result = self._generate_visualization(df, chart_config)
            
            if not visual_result['success']:
                return visual_result
            
            chart_js_data = self._convert_figure_to_chart_js_format(visual_result['chart'])
            
            return {
                'success': True,
                'chart': chart_js_data,
                'chart_type': visual_result['chart_type'],
                'configuration': chart_config,
            }
            
        except Exception as e:
            logger.error(f"Error in process_visualization_prompt: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _get_chart_configuration(self, df: pd.DataFrame, prompt: str) -> Dict[str, Any]:
        """Use NLP to determine appropriate chart type and columns with enhanced error handling"""
        try:
            # Sanitize prompt for safety
            if not prompt or len(prompt.strip()) == 0:
                return {
                    'success': False,
                    'error': 'Empty prompt provided'
                }
            
            # Limit prompt length for safety
            if len(prompt) > 5000:
                prompt = prompt[:5000]
                logger.warning("Prompt truncated to 5000 characters")
            
            # If prompt only contains chart type, add logic for default parameters
            chart_type_only = re.match(r'^\s*(line|bar|scatter|pie|histogram|box)\s*(chart)?\s*$', prompt.lower())
            
            if chart_type_only:
                requested_chart = chart_type_only.group(1).lower()
                logger.info(f"User requested just a {requested_chart} chart, using default parameters")
                
                # Use cached column types for performance
                column_types = self._get_column_types(df)
                numeric_cols = column_types['numeric']
                categorical_cols = column_types['categorical']
                date_cols = column_types['datetime']
                
                # Set defaults based on chart type and available columns
                if requested_chart == 'line':
                    x_col = date_cols[0] if date_cols else (categorical_cols[0] if categorical_cols else numeric_cols[0] if numeric_cols else df.columns[0])
                    y_col = numeric_cols[0] if numeric_cols else df.columns[0]
                    return {
                        'success': True,
                        'chart_type': 'line',
                        'x_column': x_col,
                        'y_column': y_col,
                        'title': f"{y_col} by {x_col}",
                        'aggregation': 'sum' if categorical_cols else None,
                        'reason': "Default line chart with best available parameters"
                    }
                elif requested_chart == 'bar':
                    x_col = categorical_cols[0] if categorical_cols else (date_cols[0] if date_cols else df.columns[0])
                    y_col = numeric_cols[0] if numeric_cols else (df.columns[1] if len(df.columns) > 1 else df.columns[0])
                    return {
                        'success': True,
                        'chart_type': 'bar',
                        'x_column': x_col,
                        'y_column': y_col,
                        'title': f"{y_col} by {x_col}",
                        'aggregation': 'sum',
                        'reason': "Default bar chart with best available parameters"
                    }
                elif requested_chart == 'scatter':
                    if len(numeric_cols) >= 2:
                        return {
                            'success': True,
                            'chart_type': 'scatter',
                            'x_column': numeric_cols[0],
                            'y_column': numeric_cols[1],
                            'title': f"{numeric_cols[1]} vs {numeric_cols[0]}",
                            'reason': "Default scatter plot with two numeric columns"
                        }
                    else:
                        return {
                            'success': False,
                            'error': "Scatter plot requires at least two numeric columns"
                        }
                elif requested_chart == 'pie':
                    if categorical_cols and numeric_cols:
                        return {
                            'success': True,
                            'chart_type': 'pie',
                            'x_column': categorical_cols[0],
                            'y_column': numeric_cols[0],
                            'title': f"Distribution of {numeric_cols[0]} by {categorical_cols[0]}",
                            'aggregation': 'sum',
                            'reason': "Default pie chart with categorical and numeric columns"
                        }
                    else:
                        return {
                            'success': False,
                            'error': "Pie chart requires both categorical and numeric columns"
                        }
                elif requested_chart == 'histogram':
                    if numeric_cols:
                        return {
                            'success': True,
                            'chart_type': 'histogram',
                            'x_column': numeric_cols[0],
                            'title': f"Distribution of {numeric_cols[0]}",
                            'reason': "Default histogram with numeric column"
                        }
                    else:
                        return {
                            'success': False,
                            'error': "Histogram requires at least one numeric column"
                        }
                elif requested_chart == 'box':
                    if numeric_cols:
                        return {
                            'success': True,
                            'chart_type': 'box',
                            'y_column': numeric_cols[0],
                            'x_column': categorical_cols[0] if categorical_cols else None,
                            'title': f"Box Plot of {numeric_cols[0]}",
                            'reason': "Default box plot with numeric column"
                        }
                    else:
                        return {
                            'success': False,
                            'error': "Box plot requires at least one numeric column"
                        }
            
            # Check for year-over-year comparisons with enhanced detection
            yoy_patterns = [
                r'(?:year[ -]over[ -]year|yoy|year[ -]to[ -]year|year[ -]by[ -]year)',
                r'(?:compare|comparison).*(?:current|this).*year.*(?:previous|last).*year',
                r'(?:compare|comparison).*(?:previous|last).*year.*(?:current|this).*year',
                r'(?:compare|comparison).*(\d{4}).*(\d{4})',
                r'(?:compare|comparison).*(?:sales|revenue|profit|income|profits).*(?:year|annual)',
                r'(?:compare|comparison).*(?:current|this|previous|last).*year',
                r'(?:compare|comparison).*(?:between|of|in).*(\d{4}).*(?:and|with|to).*(\d{4})',
                r'(\d{4}).*(?:versus|vs\.?|against|and|to).*(\d{4})',
                r'(?:sales|revenue|profit|income|profits).*(?:current|this).*(?:previous|last).*year',
                r'(?:sales|revenue|profit|income|profits).*(?:previous|last).*(?:current|this).*year',
                r'(?:current|this).*(?:previous|last).*year.*(?:sales|revenue|profit|income|profits)',
                r'(?:previous|last).*(?:current|this).*year.*(?:sales|revenue|profit|income|profits)'
            ]
            
            is_yoy_comparison = any(re.search(pattern, prompt, re.IGNORECASE) for pattern in yoy_patterns)
            
            # Parse time information for potential year comparison
            time_info = self._parse_time_period(prompt, df)
            
            # If this is a year-over-year comparison, handle it specially
            if is_yoy_comparison or 'compare_years' in time_info:
                logger.info(f"Detected year-over-year comparison request: {prompt}")
                
                # Use cached column types
                column_types = self._get_column_types(df)
                numeric_cols = column_types['numeric']
                date_cols = column_types['datetime']
                
                # Extract possible metric terms from prompt
                metric_terms = ['sales', 'revenue', 'profit', 'income', 'cost', 'expense', 'margin', 'volume', 'amount']
                
                # Find best matching column for the metric
                metric_col = None
                for term in metric_terms:
                    if term in prompt.lower():
                        # Find columns that contain this term
                        matches = [col for col in numeric_cols if term in col.lower()]
                        if matches:
                            metric_col = matches[0]
                            break
                
                if not metric_col and numeric_cols:
                    # Use first numeric column as fallback
                    metric_col = numeric_cols[0]
                
                # Find date column
                date_col = None
                if date_cols:
                    date_terms = ['date', 'time', 'day', 'month', 'year']
                    for term in date_terms:
                        matches = [col for col in date_cols if term in col.lower()]
                        if matches:
                            date_col = matches[0]
                            break
                    
                    if not date_col:
                        date_col = date_cols[0]
                
                # Determine if we should group by month
                group_by_month = 'month' in prompt.lower() or 'monthly' in prompt.lower()
                
                # Create a configuration for YoY chart
                chart_type = 'bar' if 'bar' in prompt.lower() else 'line'
                
                # Extract years if available
                compare_years = time_info.get('compare_years', [])
                
                # Create and return the config dict
                config = {
                    'success': True,
                    'chart_type': chart_type,
                    'x_column': date_col,
                    'y_column': metric_col,
                    'title': f"Year-over-Year Comparison of {metric_col}" if metric_col else "Year-over-Year Comparison",
                    'aggregation': 'sum',
                    'by': 'month' if group_by_month else None,
                    'is_yoy_comparison': True,
                    'time_info': time_info,
                    'compare_years': compare_years,
                    'reason': "Year-over-Year comparison chart"
                }
                
                # Make sure to convert any timestamp objects to strings before returning
                config = self.convert_timestamps_to_strings(config)
                return config
            
            # If not just a chart type, use NLP to determine configuration
            system_prompt = f"""
            You are a data visualization AI expert. Based on the natural language prompt and dataset information,
            determine the most appropriate chart type and columns to visualize the data.
            
            DataFrame Information:
            - Columns: {df.columns.tolist()}
            - Data Types: {df.dtypes.astype(str).to_dict()}
            - Sample Data: {df.head(3).to_dict()}
            
            Available Chart Types:
            - line: For trends over time or continuous variables
            - bar: For comparing quantities across categories
            - scatter: For showing relationships between two variables
            - pie: For showing proportions of a whole
            - histogram: For showing distributions of a single variable
            - box: For showing statistical distributions with outliers
            - combo: For combining bar and line charts to display related metrics
            
            User Prompt: {prompt}
            
            IMPORTANT INSTRUCTIONS:
            1. If the user mentions a specific date range (like "from 2018 to 2020", "from Jan 2016 to August 2016"), include this complete date range information in filter_condition.
            2. If the user mentions a specific time period (like "in 2023", "for Jan 2022"), include this as a filter_condition.
            3. If the user mentions "by month" or "monthly", set "by" to "month" in your response.
            4. For date columns, suggest appropriate aggregations like "month" or "year" when needed.
            5. For pie charts, ensure you select categorical data for x_column and numeric data for y_column.
            6. Explicitly identify if aggregation (sum, mean, count) should be used based on the prompt.
            7. If the user mentions comparing years, current year, previous year, or year-over-year, add "is_yoy_comparison": true to your response.
            
            Return ONLY a JSON object with the following keys:
            - chart_type: The recommended chart type (one of the options above)
            - x_column: The column name for the x-axis
            - y_column: The column name for the y-axis (can be multiple for some charts, separate by comma)
            - bar_columns: For combo charts only, columns to display as bars (comma-separated)
            - line_columns: For combo charts only, columns to display as lines (comma-separated)
            - use_secondary_y: For combo charts only, whether to use secondary y-axis (true/false)
            - title: A descriptive title for the chart
            - aggregation (optional): Aggregation method (sum, mean, count)
            - filter_condition (optional): Any filtering condition - for date ranges, include the FULL TEXT of the date range (e.g., "from 2018 to 2020", "from Jan 2016 to August 2016", "in Jan 2016")
            - by (optional): Temporal grouping like "month", "year", "day"
            - is_yoy_comparison (optional): Set to true if this is a year-over-year comparison
            - reason: A brief explanation of why this chart type is appropriate

            For COMBO charts, please specify both bar_columns and line_columns
            
            Return the JSON object only, without any markdown formatting or additional text.
            """
            
            # FIXED: Enhanced error handling for chat completion
            try:
                response = self.chat_module.chat_completion(system_prompt)
            except Exception as e:
                logger.error(f"Chat module error: {str(e)}")
                return {
                    'success': False,
                    'error': f"Chat module failed: {str(e)}"
                }
            
            # Handle different response formats with comprehensive error handling
            config_text = None
            try:
                if hasattr(response, 'choices') and len(response.get('choices', [])) > 0:
                    config_text = response['choices'][0]['message']['content']
                elif isinstance(response, dict) and 'content' in response:
                    config_text = response['content']
                elif isinstance(response, str):
                    config_text = response
                else:
                    logger.error(f"Unknown response format: {type(response)}")
                    return {'success': False, 'error': "Unable to parse chat response format"}
            except (KeyError, IndexError, TypeError) as e:
                logger.error(f"Error accessing response structure: {str(e)}")
                return {'success': False, 'error': f"Response parsing error: {str(e)}"}
            
            if not config_text:
                return {
                    'success': False,
                    'error': "Empty response from chat module"
                }
            
            # Clean up the response to ensure valid JSON
            config_text = re.sub(r'```json|```', '', config_text).strip()
            
            try:
                config = json.loads(config_text)
                
                # Ensure y_column is a list if comma-separated string was provided
                if 'y_column' in config and isinstance(config['y_column'], str) and ',' in config['y_column']:
                    config['y_column'] = [col.strip() for col in config['y_column'].split(',')]
                
                # Basic validation
                if 'chart_type' not in config or config['chart_type'] not in self.chart_types:
                    return {
                        'success': False,
                        'error': f"Invalid chart type: {config.get('chart_type', 'unknown')}"
                    }
                
                # For charts that require x and y columns
                required_xy_charts = ['line', 'bar', 'scatter']
                if config['chart_type'] in required_xy_charts:
                    if 'x_column' not in config or 'y_column' not in config:
                        return {
                            'success': False,
                            'error': f"Missing required columns for {config['chart_type']} chart"
                        }
                    
                    # Validate column names exist in the dataframe
                    if config['x_column'] not in df.columns:
                        return {
                            'success': False,
                            'error': f"Column not found: {config['x_column']}"
                        }
                    
                    if isinstance(config['y_column'], list):
                        for col in config['y_column']:
                            if col not in df.columns:
                                return {
                                    'success': False,
                                    'error': f"Column not found: {col}"
                                }
                    elif config['y_column'] not in df.columns:
                        return {
                            'success': False,
                            'error': f"Column not found: {config['y_column']}"
                        }
                
                # Get time info from filter_condition if present
                if 'filter_condition' in config and config['filter_condition']:
                    time_info = self._parse_time_period(config['filter_condition'], df)
                    
                    # If YoY comparison is detected from the filter or prompt
                    if config.get('is_yoy_comparison', False) or 'compare_years' in time_info:
                        config['is_yoy_comparison'] = True
                        config['time_info'] = time_info
                
                # Extract year from prompt if not already in filter_condition
                if 'filter_condition' not in config:
                    year_match = re.search(r'(?:in|for|during)\s+(20\d{2})', prompt, re.IGNORECASE)
                    if year_match:
                        config['filter_condition'] = f"year == {year_match.group(1)}"
                
                # Set default aggregation for charts that typically need it
                if config['chart_type'] in ['bar', 'pie'] and 'aggregation' not in config:
                    config['aggregation'] = 'sum'
                    
                # Set default 'by' for time-based visualization requests
                if 'month' in prompt.lower() and 'by' not in config:
                    config['by'] = 'month'
                    
                config['success'] = True

                config = self.convert_timestamps_to_strings(config)
                return config
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {str(e)}, content: {config_text[:500]}")
                return {
                    'success': False,
                    'error': f"Invalid JSON configuration: {str(e)}"
                }
                
        except Exception as e:
            logger.error(f"Error in _get_chart_configuration: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                'success': False,
                'error': str(e)
            }

    def _generate_visualization(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate visualization based on configuration with enhanced error handling"""
        try:
            chart_type = config['chart_type']
            is_yoy = config.get('is_yoy_comparison', False)
            
            if is_yoy and chart_type != 'yoy':
                logger.info(f"YoY comparison detected, changing chart type from {chart_type} to 'yoy'")
                config['chart_type_original'] = chart_type  # Save the original chart type
                chart_type = 'yoy'
                config['chart_type'] = 'yoy'
            
            if chart_type not in self.chart_types:
                return {
                    'success': False,
                    'error': f"Unsupported chart type: {chart_type}"
                }
            
            # Preprocess data
            processed_df = self._preprocess_data(df, config)
            
            # Validate processed data
            if processed_df.empty:
                return {
                    'success': False,
                    'error': "No data remaining after preprocessing"
                }
            
            # Get chart function and create chart with error handling
            chart_function = self.chart_types[chart_type]
            chart = self._create_chart_with_error_handling(chart_function, processed_df, config, chart_type)
            
            return {
                'success': True,
                'chart': chart,
                'chart_type': chart_type
            }
            
        except Exception as e:
            logger.error(f"Error in _generate_visualization: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _create_bar_chart(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create a bar chart using Plotly with comprehensive validation"""
        try:
            # FIXED: Add comprehensive validation
            error_msg = self._validate_config_and_data(df, config, ['x_column', 'y_column'])
            if error_msg:
                return go.Figure().add_annotation(
                    text=error_msg,
                    showarrow=False,
                    font=dict(size=14, color="red"),
                    x=0.5, y=0.5, xref="paper", yref="paper"
                )
            
            orientation = config.get('orientation', 'v')
            
            # Handle multiple y columns for grouped bars
            if isinstance(config['y_column'], list) and len(config['y_column']) > 1:
                fig = go.Figure()
                for y_col in config['y_column']:
                    if y_col not in df.columns:
                        continue
                    fig.add_trace(
                        go.Bar(
                            x=df[config['x_column']] if orientation == 'v' else df[y_col],
                            y=df[y_col] if orientation == 'v' else df[config['x_column']],
                            name=y_col.replace('_', ' ').title(),
                            orientation=orientation
                        )
                    )
                
                # Set barmode to group for multiple columns
                fig.update_layout(barmode='group')
            else:
                # Single y column
                y_col = config['y_column'][0] if isinstance(config['y_column'], list) else config['y_column']
                
                # Validate single column exists
                if y_col not in df.columns:
                    return go.Figure().add_annotation(
                        text=f"Column '{y_col}' not found in data",
                        showarrow=False,
                        font=dict(size=14, color="red"),
                        x=0.5, y=0.5, xref="paper", yref="paper"
                    )
                
                if orientation == 'h':
                    fig = px.bar(
                        df, 
                        x=y_col,
                        y=config['x_column'],
                        color=config.get('color_column'),
                        title=config.get('title', 'Bar Chart'),
                        orientation='h'
                    )
                else:
                    fig = px.bar(
                        df, 
                        x=config['x_column'],
                        y=y_col,
                        color=config.get('color_column'),
                        title=config.get('title', 'Bar Chart')
                    )

            # Format axes
            x_title = config['x_column'].replace('_', ' ').title()
            y_title = (config['y_column'][0] if isinstance(config['y_column'], list) else config['y_column']).replace('_', ' ').title()
            
            if orientation == 'h':
                # Swap for horizontal
                x_title, y_title = y_title, x_title
            
            fig.update_layout(
                title=config.get('title', 'Bar Chart'),
                xaxis_title=x_title,
                yaxis_title=y_title,
                legend_title="Legend",
                height=500
            )
            
            # Handle date axis formatting
            if orientation == 'v' and any(term in config['x_column'].lower() for term in ['date', 'month', 'year']):
                fig.update_xaxes(tickangle=45)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error in _create_bar_chart: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return go.Figure().add_annotation(
                text=f"Error creating bar chart: {str(e)}",
                showarrow=False,
                font=dict(size=14, color="red"),
                x=0.5, y=0.5, xref="paper", yref="paper"
            )
    
    def _create_line_chart(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create a line chart using Plotly with validation"""
        try:
            # FIXED: Add comprehensive validation
            error_msg = self._validate_config_and_data(df, config, ['x_column', 'y_column'])
            if error_msg:
                return go.Figure().add_annotation(
                    text=error_msg,
                    showarrow=False,
                    font=dict(size=14, color="red"),
                    x=0.5, y=0.5, xref="paper", yref="paper"
                )
            
            # Prepare data structure for multiple or single y columns
            if isinstance(config['y_column'], list) and len(config['y_column']) > 1:
                fig = go.Figure()
                for y_col in config['y_column']:
                    if y_col not in df.columns:
                        continue
                    fig.add_trace(
                        go.Scatter(
                            x=df[config['x_column']], 
                            y=df[y_col],
                            mode='lines+markers',
                            name=y_col.replace('_', ' ').title()
                        )
                    )
            else:
                y_col = config['y_column'][0] if isinstance(config['y_column'], list) else config['y_column']
                
                # Validate single column exists
                if y_col not in df.columns:
                    return go.Figure().add_annotation(
                        text=f"Column '{y_col}' not found in data",
                        showarrow=False,
                        font=dict(size=14, color="red"),
                        x=0.5, y=0.5, xref="paper", yref="paper"
                    )
                
                fig = px.line(
                    df, 
                    x=config['x_column'], 
                    y=y_col,
                    color=config.get('color_column'),
                    title=config.get('title', 'Line Chart'),
                    markers=True
                )

            # Format axis titles
            x_title = config['x_column'].replace('_', ' ').title()
            y_title = (config['y_column'][0] if isinstance(config['y_column'], list) else config['y_column']).replace('_', ' ').title()
            
            fig.update_layout(
                title=config.get('title', 'Line Chart'),
                xaxis_title=x_title,
                yaxis_title=y_title,
                legend_title="Legend",
                height=500
            )
            
            # Handle date axis formatting
            if any(term in config['x_column'].lower() for term in ['date', 'time', 'month', 'year']):
                fig.update_xaxes(
                    tickangle=45,
                    tickformat="%b %Y" if 'month' in config['x_column'].lower() else "%Y-%m-%d"
                )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error in _create_line_chart: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return go.Figure().add_annotation(
                text=f"Error creating line chart: {str(e)}",
                showarrow=False,
                font=dict(size=14, color="red"),
                x=0.5, y=0.5, xref="paper", yref="paper"
            )
    
    def _create_pie_chart(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create a pie chart using Plotly with validation"""
        try:
            # FIXED: Add comprehensive validation
            error_msg = self._validate_config_and_data(df, config, ['x_column', 'y_column'])
            if error_msg:
                return go.Figure().add_annotation(
                    text=error_msg,
                    showarrow=False,
                    font=dict(size=14, color="red"),
                    x=0.5, y=0.5, xref="paper", yref="paper"
                )
            
            values_col = config['y_column'][0] if isinstance(config['y_column'], list) else config['y_column']
            names_col = config['x_column']
            
            # Validate data types
            if not pd.api.types.is_numeric_dtype(df[values_col]):
                return go.Figure().add_annotation(
                    text=f"Values column '{values_col}' must be numeric for pie chart",
                    showarrow=False,
                    font=dict(size=14, color="red"),
                    x=0.5, y=0.5, xref="paper", yref="paper"
                )
            
            # First, ensure we're working with aggregated data
            # Group by the category column and sum (or use specified aggregation) the values
            agg_method = config.get('aggregation', 'sum')
            try:
                if agg_method == 'sum':
                    agg_df = df.groupby(names_col)[values_col].sum().reset_index()
                elif agg_method == 'mean':
                    agg_df = df.groupby(names_col)[values_col].mean().reset_index()
                elif agg_method == 'count':
                    agg_df = df.groupby(names_col)[values_col].count().reset_index()
                else:
                    agg_df = df.groupby(names_col)[values_col].sum().reset_index()
            except Exception as e:
                return go.Figure().add_annotation(
                    text=f"Error aggregating data: {str(e)}",
                    showarrow=False,
                    font=dict(size=14, color="red"),
                    x=0.5, y=0.5, xref="paper", yref="paper"
                )
            
            # If too many categories, limit to top N and group the rest
            if agg_df[names_col].nunique() > 10:
                # Sort by value and get top 9
                sorted_df = agg_df.sort_values(by=values_col, ascending=False)
                top_df = sorted_df.head(9)
                
                # Create "Other" category with the rest
                other_value = sorted_df.iloc[9:][values_col].sum() if len(sorted_df) > 9 else 0
                if other_value > 0:
                    other_df = pd.DataFrame({
                        names_col: ['Other'],
                        values_col: [other_value]
                    })
                    # Combine top categories with "Other"
                    plot_df = pd.concat([top_df, other_df])
                else:
                    plot_df = top_df
            else:
                plot_df = agg_df
            
            # Handle edge case where all values are zeros or NaN
            if plot_df[values_col].sum() == 0 or plot_df[values_col].isna().all():
                return go.Figure().add_annotation(
                    text="Cannot create pie chart: All values are zero or missing",
                    showarrow=False,
                    font=dict(size=14, color="orange"),
                    x=0.5, y=0.5, xref="paper", yref="paper"
                )
                
            fig = px.pie(
                plot_df, 
                names=names_col, 
                values=values_col,
                title=config.get('title', 'Pie Chart')
            )
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=500)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error in _create_pie_chart: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return go.Figure().add_annotation(
                text=f"Error creating pie chart: {str(e)}",
                showarrow=False,
                font=dict(size=14, color="red"),
                x=0.5, y=0.5, xref="paper", yref="paper"
            )
    
    def _create_scatter_chart(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create a scatter chart using Plotly with validation"""
        try:
            # FIXED: Add comprehensive validation
            error_msg = self._validate_config_and_data(df, config, ['x_column', 'y_column'])
            if error_msg:
                return go.Figure().add_annotation(
                    text=error_msg,
                    showarrow=False,
                    font=dict(size=14, color="red"),
                    x=0.5, y=0.5, xref="paper", yref="paper"
                )
            
            y_col = config['y_column'][0] if isinstance(config['y_column'], list) else config['y_column']
            
            # Validate both columns are numeric for meaningful scatter plot
            if not pd.api.types.is_numeric_dtype(df[config['x_column']]):
                return go.Figure().add_annotation(
                    text=f"X column '{config['x_column']}' should be numeric for scatter plot",
                    showarrow=False,
                    font=dict(size=14, color="orange"),
                    x=0.5, y=0.5, xref="paper", yref="paper"
                )
                
            if not pd.api.types.is_numeric_dtype(df[y_col]):
                return go.Figure().add_annotation(
                    text=f"Y column '{y_col}' should be numeric for scatter plot",
                    showarrow=False,
                    font=dict(size=14, color="orange"),
                    x=0.5, y=0.5, xref="paper", yref="paper"
                )
            
            fig = px.scatter(
                df, 
                x=config['x_column'], 
                y=y_col,
                color=config.get('color_column'),
                size=config.get('size_column'),
                hover_data=df.columns,
                title=config.get('title', 'Scatter Plot')
            )
            
            fig.update_layout(
                xaxis_title=config['x_column'].replace('_', ' ').title(),
                yaxis_title=y_col.replace('_', ' ').title(),
                legend_title="Legend",
                height=500
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error in _create_scatter_chart: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return go.Figure().add_annotation(
                text=f"Error creating scatter chart: {str(e)}",
                showarrow=False,
                font=dict(size=14, color="red"),
                x=0.5, y=0.5, xref="paper", yref="paper"
            )
    
    def _create_histogram(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create a histogram using Plotly with validation"""
        try:
            # FIXED: Add validation for histogram
            if 'x_column' not in config:
                return go.Figure().add_annotation(
                    text="Missing x_column for histogram",
                    showarrow=False,
                    font=dict(size=14, color="red"),
                    x=0.5, y=0.5, xref="paper", yref="paper"
                )
            
            values_col = config['x_column']
            
            if values_col not in df.columns:
                return go.Figure().add_annotation(
                    text=f"Column '{values_col}' not found in data",
                    showarrow=False,
                    font=dict(size=14, color="red"),
                    x=0.5, y=0.5, xref="paper", yref="paper"
                )
            
            # Validate data type
            if not pd.api.types.is_numeric_dtype(df[values_col]):
                return go.Figure().add_annotation(
                    text=f"Column '{values_col}' must be numeric for histogram",
                    showarrow=False,
                    font=dict(size=14, color="orange"),
                    x=0.5, y=0.5, xref="paper", yref="paper"
                )
            
            # Automatically determine bin count based on data
            n_bins = min(25, max(5, int(df[values_col].nunique() / 2)))
            
            fig = px.histogram(
                df, 
                x=values_col,
                color=config.get('color_column'),
                nbins=n_bins,
                title=config.get('title', 'Histogram'),
                opacity=0.8,
                histnorm=config.get('histnorm', None)
            )
            
            fig.update_layout(
                xaxis_title=values_col.replace('_', ' ').title(),
                yaxis_title="Count",
                bargap=0.05,
                height=500
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error in _create_histogram: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return go.Figure().add_annotation(
                text=f"Error creating histogram: {str(e)}",
                showarrow=False,
                font=dict(size=14, color="red"),
                x=0.5, y=0.5, xref="paper", yref="paper"
            )
    
    def _create_box_plot(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create a box plot using Plotly with validation"""
        try:
            # FIXED: Add validation for box plot
            if 'y_column' not in config:
                return go.Figure().add_annotation(
                    text="Missing y_column for box plot",
                    showarrow=False,
                    font=dict(size=14, color="red"),
                    x=0.5, y=0.5, xref="paper", yref="paper"
                )
                
            # Handle multiple y columns
            if isinstance(config['y_column'], list) and len(config['y_column']) > 1:
                fig = go.Figure()
                for col in config['y_column']:
                    if col not in df.columns:
                        continue
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        continue
                    fig.add_trace(go.Box(
                        y=df[col],
                        name=col.replace('_', ' ').title(),
                        boxmean=True  # Show mean as a dashed line
                    ))
                    
                fig.update_layout(
                    title=config.get('title', 'Box Plot'),
                    yaxis_title="Value",
                    showlegend=True,
                    height=500
                )
            else:
                y_col = config['y_column'][0] if isinstance(config['y_column'], list) else config['y_column']
                
                if y_col not in df.columns:
                    return go.Figure().add_annotation(
                        text=f"Column '{y_col}' not found in data",
                        showarrow=False,
                        font=dict(size=14, color="red"),
                        x=0.5, y=0.5, xref="paper", yref="paper"
                    )
                
                # Validate data type
                if not pd.api.types.is_numeric_dtype(df[y_col]):
                    return go.Figure().add_annotation(
                        text=f"Column '{y_col}' must be numeric for box plot",
                        showarrow=False,
                        font=dict(size=14, color="orange"),
                        x=0.5, y=0.5, xref="paper", yref="paper"
                    )
                
                fig = px.box(
                    df, 
                    x=config.get('x_column'),
                    y=y_col,
                    color=config.get('color_column'),
                    title=config.get('title', 'Box Plot'),
                    points="outliers"  # Show only outliers as individual points
                )
                
                fig.update_layout(
                    xaxis_title=config.get('x_column', '').replace('_', ' ').title() if config.get('x_column') else '',
                    yaxis_title=y_col.replace('_', ' ').title(),
                    height=500
                )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error in _create_box_plot: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return go.Figure().add_annotation(
                text=f"Error creating box plot: {str(e)}",
                showarrow=False,
                font=dict(size=14, color="red"),
                x=0.5, y=0.5, xref="paper", yref="paper"
            )
    
    def _create_combo_chart(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create a combination chart with bars and lines using Plotly with validation"""
        try:
            # FIXED: Add validation for combo chart
            if 'x_column' not in config:
                return go.Figure().add_annotation(
                    text="Missing x_column for combo chart",
                    showarrow=False,
                    font=dict(size=14, color="red"),
                    x=0.5, y=0.5, xref="paper", yref="paper"
                )
            
            fig = go.Figure()
            
            # Get columns for bars and lines
            bar_columns = config.get('bar_columns', [])
            line_columns = config.get('line_columns', [])
            
            # Convert string to list if needed
            if isinstance(bar_columns, str):
                bar_columns = [col.strip() for col in bar_columns.split(',')]
            if isinstance(line_columns, str):
                line_columns = [col.strip() for col in line_columns.split(',')]
            
            # Validate that we have columns to plot
            if not bar_columns and not line_columns:
                return go.Figure().add_annotation(
                    text="Combo chart requires either bar_columns or line_columns",
                    showarrow=False,
                    font=dict(size=14, color="red"),
                    x=0.5, y=0.5, xref="paper", yref="paper"
                )
                
            # Add bar traces
            for i, col in enumerate(bar_columns):
                if col in df.columns:
                    fig.add_trace(
                        go.Bar(
                            x=df[config['x_column']],
                            y=df[col],
                            name=col.replace('_', ' ').title(),
                            yaxis='y',
                            marker_color=f'rgba(54, 162, 235, {0.7 - (i * 0.1)})'  # Blue with decreasing opacity
                        )
                    )
            
            # Add line traces
            for i, col in enumerate(line_columns):
                if col in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df[config['x_column']],
                            y=df[col],
                            mode='lines+markers',
                            name=col.replace('_', ' ').title(),
                            yaxis='y2' if config.get('use_secondary_y', False) else 'y',
                            marker_color=f'rgba(255, 99, 132, {0.9 - (i * 0.1)})'  # Red with decreasing opacity
                        )
                    )
            
            # Format axis titles
            x_title = config['x_column'].replace('_', ' ').title()
            
            # Update layout with secondary y-axis if needed
            if config.get('use_secondary_y', False) and line_columns:
                fig.update_layout(
                    title=config.get('title', 'Combo Chart'),
                    xaxis_title=x_title,
                    yaxis=dict(
                        title=f"{' & '.join(bar_columns).replace('_', ' ').title()}",
                        side="left"
                    ),
                    yaxis2=dict(
                        title=f"{' & '.join(line_columns).replace('_', ' ').title()}",
                        side="right",
                        overlaying="y",
                        showgrid=False
                    ),
                    legend_title="Legend",
                    height=500,
                    barmode='group'
                )
            else:
                fig.update_layout(
                    title=config.get('title', 'Combo Chart'),
                    xaxis_title=x_title,
                    yaxis_title="Values",
                    legend_title="Legend",
                    height=500,
                    barmode='group'
                )
            
            # Handle date axis formatting
            if any(term in config['x_column'].lower() for term in ['date', 'time', 'month', 'year']):
                fig.update_xaxes(
                    tickangle=45,
                    tickformat="%b %Y" if 'month' in config['x_column'].lower() else "%Y-%m-%d"
                )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error in _create_combo_chart: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return go.Figure().add_annotation(
                text=f"Error creating combo chart: {str(e)}",
                showarrow=False,
                font=dict(size=14, color="red"),
                x=0.5, y=0.5, xref="paper", yref="paper"
            )
    
    def _create_yoy_chart(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create a year-over-year comparison chart using Plotly with validation"""
        try:
            # FIXED: Add validation for YoY chart
            error_msg = self._validate_config_and_data(df, config, ['y_column'])
            if error_msg:
                return go.Figure().add_annotation(
                    text=error_msg,
                    showarrow=False,
                    font=dict(size=14, color="red"),
                    x=0.5, y=0.5, xref="paper", yref="paper"
                )
                
            # Determine if we should use line or bar chart based on config
            original_chart_type = config.get('chart_type_original', 'line')
            is_bar = original_chart_type == 'bar' or 'bar' in config.get('title', '').lower()
            
            # Get the metric column to visualize
            y_col = config['y_column']
            if isinstance(y_col, list):
                y_col = y_col[0]  # Use the first column if multiple are provided
            
            # Get the x and color columns
            x_col = config.get('x_column', 'MonthName')
            color_col = config.get('color_column', 'Year')
            
            # Validate columns exist
            if x_col and x_col not in df.columns:
                logger.warning(f"X column '{x_col}' not found, will use available columns")
                # Try to find a suitable x column
                if 'MonthName' in df.columns:
                    x_col = 'MonthName'
                elif 'Month' in df.columns:
                    x_col = 'Month'
                else:
                    x_col = df.columns[0]  # Fallback to first column
            
            if color_col not in df.columns:
                logger.warning(f"Color column '{color_col}' not found, will use available columns")
                # Try to find a suitable color column
                if 'Year' in df.columns:
                    color_col = 'Year'
                else:
                    # Create a simple year column if none exists
                    df['Year'] = '2023'  # Default value
                    color_col = 'Year'
            
            # Check if data is grouped by month
            by_month = config.get('by') == 'month' or 'Month' in df.columns or 'MonthName' in df.columns
            
            # Sort data if using months
            if by_month and 'Month' in df.columns:
                df = df.sort_values('Month')
            
            # Create chart title based on the data
            years = df[color_col].unique()
            years_str = " vs ".join(sorted([str(y) for y in years]))
            metric_name = y_col.replace('_', ' ').title()
            
            if by_month:
                title = f"{metric_name} Comparison: {years_str} (Monthly)"
            else:
                title = f"{metric_name} Comparison: {years_str}"
                
            # Override with user-specified title if provided
            if 'title' in config and config['title']:
                title = config['title']
            
            # Create the figure
            if is_bar:
                fig = px.bar(
                    df,
                    x=x_col,
                    y=y_col,
                    color=color_col,
                    title=title,
                    barmode='group'
                )
            else:
                fig = px.line(
                    df,
                    x=x_col,
                    y=y_col,
                    color=color_col,
                    title=title,
                    markers=True
                )
            
            # Format axis titles
            x_title = x_col.replace('_', ' ').title()
            y_title = y_col.replace('_', ' ').title()
            
            # Update layout
            fig.update_layout(
                xaxis_title=x_title,
                yaxis_title=y_title,
                legend_title="Year",
                height=500
            )
            
            # For month-based x-axis, customize the order
            if by_month:
                # Create a custom month order
                month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                
                # Check if the x column matches our expected MonthName format
                if x_col == 'MonthName' and len(df) > 0 and df[x_col].iloc[0] in month_order:
                    fig.update_xaxes(categoryorder='array', categoryarray=month_order)
                
                # Add grid lines for better readability
                fig.update_xaxes(showgrid=True)
                fig.update_yaxes(showgrid=True)
            
            # Add annotations for insights
            years = df[color_col].unique()
            if len(years) == 2 and isinstance(years[0], str) and isinstance(years[1], str):
                try:
                    # Calculate year-over-year growth rate
                    year1_total = df[df[color_col] == years[0]][y_col].sum()
                    year2_total = df[df[color_col] == years[1]][y_col].sum()
                    
                    if year1_total > 0:  # Avoid division by zero
                        growth_rate = ((year2_total - year1_total) / year1_total) * 100
                        growth_text = f"Overall Growth: {growth_rate:.1f}%"
                        
                        # Determine color based on growth
                        growth_color = "green" if growth_rate > 0 else "red"
                        
                        fig.add_annotation(
                            x=0.5,
                            y=1.05,
                            xref="paper",
                            yref="paper",
                            text=growth_text,
                            showarrow=False,
                            font=dict(size=12, color=growth_color),
                            bgcolor="rgba(255, 255, 255, 0.8)",
                            bordercolor="gray",
                            borderwidth=1
                        )
                except Exception as e:
                    logger.warning(f"Error calculating growth rate: {str(e)}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error in _create_yoy_chart: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return go.Figure().add_annotation(
                text=f"Error creating YoY chart: {str(e)}",
                showarrow=False,
                font=dict(size=14, color="red"),
                x=0.5, y=0.5, xref="paper", yref="paper"
            )
    
    def _parse_time_period(self, prompt: str, df: pd.DataFrame = None) -> Dict[str, Any]:
        """Parse time period information from a prompt with enhanced date range and relative time support"""
        time_info = {}
        
        # Check for relative time periods first
        relative_time_info = self._parse_relative_time_periods(prompt, df)
        if relative_time_info:
            return relative_time_info
        
        # If no relative time period found, proceed with existing parsing logic
        
        # Check for date ranges (YYYY-YYYY format)
        explicit_years_pattern = r'(?:in|for|during)?\s+(?:year|years)?\s*(20\d{2})\s+(?:and|&|vs\.?|versus)\s+(20\d{2})'
        explicit_years_match = re.search(explicit_years_pattern, prompt, re.IGNORECASE)
        if explicit_years_match:
            year1 = int(explicit_years_match.group(1))
            year2 = int(explicit_years_match.group(2))
            time_info['compare_years'] = [year1, year2]
            
        # Check for specific year mentions
        elif single_year_match := re.search(r'(?:in|for|during)\s+(20\d{2})', prompt, re.IGNORECASE):
            time_info['year'] = int(single_year_match.group(1))
        
        # Check for month-year range (e.g., "from Jan 2016 to August 2016")
        month_range_match = re.search(
            r'(?:from|between)\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t)?(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(20\d{2})\s+(?:to|and|through|-)?\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t)?(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(20\d{2})',
            prompt, re.IGNORECASE
        )
        
        if month_range_match:
            # Extract the full text of the match to parse with datetime
            range_text = month_range_match.group(0)
            # Extract dates with more flexible parsing
            matches = re.findall(r'((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t)?(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+20\d{2})', range_text, re.IGNORECASE)
            
            if len(matches) >= 2:
                try:
                    time_info['start_date'] = pd.to_datetime(matches[0], format='%B %Y', errors='coerce')
                    time_info['end_date'] = pd.to_datetime(matches[1], format='%B %Y', errors='coerce')
                except:
                    # Try with abbreviated month format if full month format fails
                    try:
                        time_info['start_date'] = pd.to_datetime(matches[0], format='%b %Y', errors='coerce')
                        time_info['end_date'] = pd.to_datetime(matches[1], format='%b %Y', errors='coerce')
                    except:
                        # If parsing fails, remove this info
                        time_info.pop('start_date', None)
                        time_info.pop('end_date', None)
        
        # Check for specific month-year mention (e.g., "in Jan 2016")
        specific_month_match = re.search(
            r'(?:in|for|during)\s+((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t)?(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+20\d{2})',
            prompt, re.IGNORECASE
        )
        
        if specific_month_match:
            month_year = specific_month_match.group(1)
            try:
                time_info['specific_date'] = pd.to_datetime(month_year, format='%B %Y', errors='coerce')
            except:
                try:
                    time_info['specific_date'] = pd.to_datetime(month_year, format='%b %Y', errors='coerce')
                except:
                    # If parsing fails, remove this info
                    time_info.pop('specific_date', None)
        
        # Check for monthly grouping
        if re.search(r'by\s+month|monthly', prompt, re.IGNORECASE):
            time_info['by'] = 'month'
            
        # Check for quarterly grouping
        if re.search(r'by\s+quarter|quarterly', prompt, re.IGNORECASE):
            time_info['by'] = 'quarter'
            
        # Check for yearly grouping
        if re.search(r'by\s+year|yearly|annual', prompt, re.IGNORECASE):
            time_info['by'] = 'year'
            
        return time_info
    
    def _parse_relative_time_periods(self, prompt: str, df: pd.DataFrame = None) -> Dict[str, Any]:
        """Parse relative time periods like 'current year', 'last year', etc."""
        time_info = {}
        
        # Get reference year (from data if available, otherwise current year)
        current_year = self._get_reference_year(df) if df is not None else pd.Timestamp.now().year
        
        # Check for current year mentions
        current_year_pattern = r'(?:in\s+|for\s+|during\s+)?(?:the\s+)?(?:current|this)\s+year'
        if re.search(current_year_pattern, prompt, re.IGNORECASE):
            time_info['year'] = current_year
            time_info['is_relative'] = True
            time_info['relative_type'] = 'current_year'
            
            # Also add start and end dates for easier filtering
            time_info['start_date'] = f"{current_year}-01-01"
            time_info['end_date'] = f"{current_year}-12-31"
            
            # Check if "by month" or "monthly" is mentioned
            if re.search(r'by\s+month|monthly', prompt, re.IGNORECASE):
                time_info['by'] = 'month'
        
        # Check for previous/last year mentions
        previous_year_pattern = r'(?:in\s+|for\s+|during\s+)?(?:the\s+)?(?:previous|last)\s+year'
        if re.search(previous_year_pattern, prompt, re.IGNORECASE):
            previous_year = current_year - 1
            time_info['year'] = previous_year
            time_info['is_relative'] = True
            time_info['relative_type'] = 'previous_year'
            
            # Also add start and end dates for easier filtering
            time_info['start_date'] = f"{previous_year}-01-01"
            time_info['end_date'] = f"{previous_year}-12-31"
            
            # Check if "by month" or "monthly" is mentioned
            if re.search(r'by\s+month|monthly', prompt, re.IGNORECASE):
                time_info['by'] = 'month'
        
        # Check for comparison between years
        comparison_pattern = r'(?:compare|comparison\s+(?:between|of))\s+(?:sales|data|revenue|performance|results)?\s*(?:of|in|between|for)?\s*(?:the\s+)?(current|this)\s+year\s+(?:and|with|to)\s+(?:the\s+)?(previous|last)\s+year'
        if re.search(comparison_pattern, prompt, re.IGNORECASE):
            time_info['is_relative'] = True
            time_info['relative_type'] = 'year_comparison'
            time_info['compare_years'] = [current_year - 1, current_year]
            
            # Check if "by month" or "monthly" is mentioned
            if re.search(r'by\s+month|monthly', prompt, re.IGNORECASE):
                time_info['by'] = 'month'
        
        # Check for specific year-to-year comparison
        specific_years_comparison = re.search(r'(?:compare|comparison\s+(?:between|of))\s+(?:sales|data|revenue|performance|results)?\s*(?:of|in|between|for)?\s*(20\d{2})\s+(?:and|with|to)\s+(20\d{2})', prompt, re.IGNORECASE)
        if specific_years_comparison:
            year1 = int(specific_years_comparison.group(1))
            year2 = int(specific_years_comparison.group(2))
            time_info['is_relative'] = False
            time_info['compare_years'] = [year1, year2]
            
            # Check if "by month" or "monthly" is mentioned
            if re.search(r'by\s+month|monthly', prompt, re.IGNORECASE):
                time_info['by'] = 'month'
        
        return time_info
    
    def _preprocess_data(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Preprocess data based on configuration with enhanced error handling"""
        try:
            processed_df = df.copy()
            
            # Parse time information from filter condition
            time_info = {}
            if 'filter_condition' in config and config['filter_condition']:
                time_info = self._parse_time_period(config['filter_condition'], df)
            
            # If the configuration already has time_info, use that
            if 'time_info' in config and config['time_info']:
                time_info = config['time_info']
            
            # Special handling for year-over-year comparisons
            is_yoy = (
                config.get('is_yoy_comparison', False) or 
                config.get('chart_type', '') == 'yoy' or
                'compare_years' in time_info or
                time_info.get('relative_type') in ['year_comparison']
            )
        
            if is_yoy:
                logger.info("Processing year-over-year comparison data")
                config['is_yoy_comparison'] = True
                return self._prepare_year_comparison_data(df, config, time_info)
            
            # Apply date filtering if specified
            date_columns = self._get_date_columns(df)
            
            if date_columns:
                # Use the first date column by default
                date_col = date_columns[0]
                
                # Try to find a date column that matches config's x_column if possible
                if 'x_column' in config and config['x_column'] in date_columns:
                    date_col = config['x_column']
                
                # Check for relative time periods first (current/previous year)
                if time_info.get('is_relative', False):
                    relative_type = time_info.get('relative_type', '')
                    
                    # Handle current year filter
                    if relative_type == 'current_year' or ('year' in time_info and time_info.get('is_relative', False)):
                        year = time_info.get('year', self._get_reference_year(df))
                        processed_df = processed_df[processed_df[date_col].dt.year == year]
                        logger.info(f"Filtered data for current year {year} using {date_col}")
                    
                    # Handle previous year filter
                    elif relative_type == 'previous_year':
                        year = time_info.get('year', self._get_reference_year(df) - 1)
                        processed_df = processed_df[processed_df[date_col].dt.year == year]
                        logger.info(f"Filtered data for previous year {year} using {date_col}")
                    
                    # Handle specific date ranges
                    elif 'start_date' in time_info and 'end_date' in time_info:
                        start_date = pd.to_datetime(time_info['start_date'])
                        end_date = pd.to_datetime(time_info['end_date'])
                        processed_df = processed_df[
                            (processed_df[date_col] >= start_date) &
                            (processed_df[date_col] <= end_date)
                        ]
                        logger.info(f"Filtered data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} using {date_col}")
                
                # Non-relative time filters
                else:
                    # Case 1: Specific year filter
                    if 'year' in time_info:
                        year = time_info['year']
                        processed_df = processed_df[processed_df[date_col].dt.year == year]
                        logger.info(f"Filtered data for year {year} using {date_col}")
                        
                    # Case 2: Date range filters
                    elif 'start_date' in time_info and 'end_date' in time_info:
                        start_date = pd.to_datetime(time_info['start_date'])
                        end_date = pd.to_datetime(time_info['end_date'])
                        processed_df = processed_df[
                            (processed_df[date_col] >= start_date) &
                            (processed_df[date_col] <= end_date)
                        ]
                        logger.info(f"Filtered data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} using {date_col}")
                        
                    # Case 3: Specific month-year filter
                    elif 'specific_date' in time_info:
                        specific_date = pd.to_datetime(time_info['specific_date'])
                        next_month = specific_date + pd.DateOffset(months=1)
                        processed_df = processed_df[
                            (processed_df[date_col] >= specific_date) &
                            (processed_df[date_col] < next_month)
                        ]
                        logger.info(f"Filtered data for {specific_date.strftime('%b %Y')} using {date_col}")
            
            # Apply time-based aggregation
            if 'by' in config and config.get('by') == 'month' and date_columns:
                date_col = date_columns[0]
                if 'x_column' in config and config['x_column'] in date_columns:
                    date_col = config['x_column']
                    
                processed_df['month'] = processed_df[date_col].dt.to_period('M')
                
                # Group by month
                if 'y_column' in config:
                    if isinstance(config['y_column'], list):
                        y_cols = config['y_column']
                    else:
                        y_cols = [config['y_column']]
                        
                    # Perform aggregation by month
                    agg_method = config.get('aggregation', 'sum').lower()
                    try:
                        if agg_method == 'sum':
                            processed_df = processed_df.groupby('month')[y_cols].sum().reset_index()
                        elif agg_method == 'mean':
                            processed_df = processed_df.groupby('month')[y_cols].mean().reset_index()
                        elif agg_method == 'count':
                            processed_df = processed_df.groupby('month')[y_cols].count().reset_index()
                        
                        # Convert period back to datetime for proper charting
                        processed_df['month'] = processed_df['month'].dt.to_timestamp()
                        processed_df['month_formatted'] = processed_df['month'].dt.strftime('%b %Y')
                        
                        # Update x_column to use the formatted month column
                        config['x_column'] = 'month_formatted'
                    except Exception as e:
                        logger.warning(f"Monthly aggregation failed: {str(e)}. Using raw data.")
            
            # Apply regular aggregation if specified
            elif 'aggregation' in config and config['aggregation'] and 'x_column' in config and 'y_column' in config:
                agg_method = config['aggregation'].lower()
                x_col = config['x_column']
                
                if isinstance(config['y_column'], list):
                    y_cols = config['y_column'] 
                else:
                    y_cols = [config['y_column']]
                
                # Perform aggregation
                try:
                    if agg_method == 'sum':
                        processed_df = processed_df.groupby(x_col)[y_cols].sum().reset_index()
                    elif agg_method == 'mean':
                        processed_df = processed_df.groupby(x_col)[y_cols].mean().reset_index()
                    elif agg_method == 'count':
                        processed_df = processed_df.groupby(x_col)[y_cols].count().reset_index()
                    logger.info(f"Applied {agg_method} aggregation on {y_cols} grouped by {x_col}")
                except Exception as e:
                    logger.warning(f"Aggregation failed: {str(e)}. Proceeding with original data.")
            
            # Handle date formatting for better display
            for col in processed_df.columns:
                if pd.api.types.is_datetime64_any_dtype(processed_df[col]):
                    if col == config.get('x_column', ''):
                        if 'month' in config.get('by', '').lower():
                            processed_df[f'{col}_formatted'] = processed_df[col].dt.strftime('%b %Y')
                        else:
                            processed_df[f'{col}_formatted'] = processed_df[col].dt.strftime('%Y-%m-%d')
                        config['x_column'] = f'{col}_formatted'
                        logger.info(f"Formatted date column {col} to {col}_formatted")
            
            # Apply sorting
            if 'sort_by' in config and config['sort_by'] and config['sort_by'] in processed_df.columns:
                processed_df = processed_df.sort_values(by=config['sort_by'])
            
            # Default sort by x_column if it's a date
            elif 'x_column' in config and config['x_column'] in processed_df.columns:
                if processed_df[config['x_column']].dtype.kind in 'mM':  # datetime types
                    processed_df = processed_df.sort_values(by=config['x_column'])
            
            return processed_df
            
        except Exception as e:
            logger.error(f"Error in _preprocess_data: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return df
    
    def _prepare_year_comparison_data(self, df: pd.DataFrame, config: Dict[str, Any], time_info: Dict[str, Any]) -> pd.DataFrame:
        """Prepare data for year-to-year comparison with enhanced error handling"""
        try:
            processed_df = df.copy()
            
            # Find date column
            date_columns = self._get_date_columns(df)
            
            if not date_columns:
                logger.warning("No date columns found for year comparison")
                return processed_df
            
            # Use the first date column by default or x_column if it's a date
            date_col = date_columns[0]
            if 'x_column' in config and config['x_column'] in date_columns:
                date_col = config['x_column']
            
            # Extract the years to compare
            compare_years = []
            
            if 'compare_years' in time_info and len(time_info['compare_years']) >= 2:
                compare_years = [int(y) for y in time_info['compare_years']]
                logger.info(f"Using explicitly specified years for comparison: {compare_years}")
            else:
                # Get all available years from the dataset
                available_years = sorted(df[date_col].dt.year.unique())
                
                if len(available_years) >= 2:
                    # Use the latest two years from the dataset
                    compare_years = available_years[-2:]
                else:
                    # If only one year is available, use it and the next year
                    compare_years = [available_years[0], available_years[0] + 1] if available_years else [2022, 2023]
                logger.info(f"Using inferred years for comparison: {compare_years}")
            
            # FIXED: Use more efficient filtering
            year1_mask = df[date_col].dt.year == compare_years[0]
            year2_mask = df[date_col].dt.year == compare_years[1]
            
            year1_data = df.loc[year1_mask].copy() if year1_mask.any() else pd.DataFrame()
            year2_data = df.loc[year2_mask].copy() if year2_mask.any() else pd.DataFrame()
            
            # Handle missing data
            if year1_data.empty:
                logger.warning(f"No data found for year {compare_years[0]}")
                # Create placeholder data if needed
                if not year2_data.empty:
                    year1_data = year2_data.copy()
                    # Set all numeric values to 0
                    numeric_cols = year1_data.select_dtypes(include=[np.number]).columns
                    year1_data[numeric_cols] = 0
                    # Set the date to year1
                    year1_data[date_col] = year1_data[date_col].apply(
                        lambda x: x.replace(year=compare_years[0])
                    )
            
            if year2_data.empty:
                logger.warning(f"No data found for year {compare_years[1]}")
                # Create placeholder data if needed
                if not year1_data.empty:
                    year2_data = year1_data.copy()
                    # Set all numeric values to 0
                    numeric_cols = year2_data.select_dtypes(include=[np.number]).columns
                    year2_data[numeric_cols] = 0
                    # Set the date to year2
                    year2_data[date_col] = year2_data[date_col].apply(
                        lambda x: x.replace(year=compare_years[1])
                    )
            
            # Add year information as a string column
            year1_data['Year'] = str(compare_years[0])
            year2_data['Year'] = str(compare_years[1])
            
            # Determine aggregation level
            is_monthly = (
                time_info.get('by') == 'month' or 
                'by' in config and config['by'] == 'month' or
                'month' in str(config.get('by', '')).lower()
            )
            
            if is_monthly:
                # Extract month for comparison
                year1_data['Month'] = year1_data[date_col].dt.month
                year2_data['Month'] = year2_data[date_col].dt.month
                
                # Add month name for better readability
                year1_data['MonthName'] = year1_data[date_col].dt.strftime('%b')
                year2_data['MonthName'] = year2_data[date_col].dt.strftime('%b')
                
                # Determine which columns to aggregate
                agg_columns = []
                if 'y_column' in config:
                    if isinstance(config['y_column'], list):
                        agg_columns = config['y_column']
                    else:
                        agg_columns = [config['y_column']]
                
                # If no columns specified, use numeric columns
                if not agg_columns:
                    agg_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                    # Remove potential ID columns
                    agg_columns = [col for col in agg_columns 
                                 if not any(term in col.lower() for term in ['id', 'year', 'month'])]
                
                # Aggregate by month
                agg_method = config.get('aggregation', 'sum')
                agg_dict = {col: agg_method for col in agg_columns if col in df.columns}
                
                if agg_dict:
                    try:
                        year1_data = year1_data.groupby(['Month', 'MonthName', 'Year']).agg(agg_dict).reset_index()
                        year2_data = year2_data.groupby(['Month', 'MonthName', 'Year']).agg(agg_dict).reset_index()
                        
                        # Sort by month
                        year1_data = year1_data.sort_values('Month')
                        year2_data = year2_data.sort_values('Month')
                    except Exception as e:
                        logger.warning(f"Monthly aggregation failed: {str(e)}. Using raw data.")
            
            # Combine the data from both years
            combined_df = pd.concat([year1_data, year2_data], ignore_index=True)
            
            # Update config to use the new columns
            if is_monthly:
                config['x_column'] = 'MonthName'
                config['color_column'] = 'Year'
                config['by'] = 'month'
            else:
                config['color_column'] = 'Year'
            
            # Ensure we're using YoY chart type
            config['chart_type_original'] = config.get('chart_type', 'line')
            config['chart_type'] = 'yoy'
            
            return combined_df
            
        except Exception as e:
            logger.error(f"Error in _prepare_year_comparison_data: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return df
    
    # Utility methods for recommendations and examples
    def suggest_chart_types(self, df: pd.DataFrame, column_name: str) -> List[str]:
        """Suggest chart types based on column data type"""
        try:
            if column_name not in df.columns:
                return []
                
            data_type = df[column_name].dtype
            suggestions = []
            
            if pd.api.types.is_datetime64_any_dtype(data_type):
                suggestions = ['line', 'bar']
            elif pd.api.types.is_numeric_dtype(data_type):
                suggestions = ['histogram', 'box', 'scatter']
            elif pd.api.types.is_categorical_dtype(data_type) or pd.api.types.is_object_dtype(data_type):
                suggestions = ['bar', 'pie']
                
            return suggestions
            
        except Exception as e:
            logger.error(f"Error in suggest_chart_types: {str(e)}")
            return []
    
    def get_chart_types(self) -> Dict[str, str]:
        """Return all available chart types with descriptions"""
        return self.chart_descriptions
    
    def get_recommended_charts(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Recommend charts based on the dataframe structure"""
        try:
            recommendations = []
            
            # Use cached column types
            column_types = self._get_column_types(df)
            numeric_cols = column_types['numeric']
            categorical_cols = column_types['categorical']
            date_cols = column_types['datetime']
            
            # If we have date columns and numeric columns, suggest time series
            if date_cols and numeric_cols:
                recommendations.append({
                    'chart_type': 'line',
                    'title': f'Time Series of {numeric_cols[0]}',
                    'description': f'Track how {numeric_cols[0]} changes over time',
                    'x_column': date_cols[0],
                    'y_column': numeric_cols[0]
                })
            
            # If we have categorical and numeric columns, suggest bar charts
            if categorical_cols and numeric_cols:
                recommendations.append({
                    'chart_type': 'bar',
                    'title': f'{numeric_cols[0]} by {categorical_cols[0]}',
                    'description': f'Compare {numeric_cols[0]} across different {categorical_cols[0]} categories',
                    'x_column': categorical_cols[0],
                    'y_column': numeric_cols[0]
                })
                
                recommendations.append({
                    'chart_type': 'pie',
                    'title': f'Distribution of {numeric_cols[0]} by {categorical_cols[0]}',
                    'description': f'See the proportion of {numeric_cols[0]} for each {categorical_cols[0]}',
                    'x_column': categorical_cols[0],
                    'y_column': numeric_cols[0]
                })
            
            # If we have multiple numeric columns, suggest scatter plot
            if len(numeric_cols) >= 2:
                recommendations.append({
                    'chart_type': 'scatter',
                    'title': f'Relationship between {numeric_cols[0]} and {numeric_cols[1]}',
                    'description': f'Explore correlation between {numeric_cols[0]} and {numeric_cols[1]}',
                    'x_column': numeric_cols[0],
                    'y_column': numeric_cols[1]
                })
            
            # If we have a single numeric column, suggest histogram
            if numeric_cols:
                recommendations.append({
                    'chart_type': 'histogram',
                    'title': f'Distribution of {numeric_cols[0]}',
                    'description': f'View the distribution of {numeric_cols[0]} values',
                    'x_column': numeric_cols[0],
                    'y_column': None
                })
                
                recommendations.append({
                    'chart_type': 'box',
                    'title': f'Box Plot of {numeric_cols[0]}',
                    'description': f'Examine the statistical distribution of {numeric_cols[0]}',
                    'x_column': None,
                    'y_column': numeric_cols[0]
                })
            
            return recommendations[:5]  
            
        except Exception as e:
            logger.error(f"Error in get_recommended_charts: {str(e)}")
            return []
        
    def generate_example_prompts(self, df: pd.DataFrame) -> List[str]:
        """Generate example visualization prompts based on the dataframe structure"""
        try:
            examples = []
            
            # Use cached column types
            column_types = self._get_column_types(df)
            numeric_cols = column_types['numeric']
            categorical_cols = column_types['categorical']
            date_cols = column_types['datetime']
            
            # Time series prompts
            if date_cols and numeric_cols:
                examples.append(f"Show me trends of {numeric_cols[0]} over time")
                
            # Comparison prompts
            if categorical_cols and numeric_cols:
                examples.append(f"Create a bar chart comparing {numeric_cols[0]} across different {categorical_cols[0]}")
                examples.append(f"Show me a pie chart of {numeric_cols[0]} distribution by {categorical_cols[0]}")
                
            # Distribution prompts
            if numeric_cols:
                examples.append(f"Visualize the distribution of {numeric_cols[0]} values")
                
            # Correlation prompts
            if len(numeric_cols) >= 2:
                examples.append(f"Show me the relationship between {numeric_cols[0]} and {numeric_cols[1]}")
                
            # Add some general prompts if we don't have enough
            if len(examples) < 3:
                examples.append("Create a visualization that shows the most important patterns in the data")
                examples.append("What's the best way to visualize the key trends in this dataset?")
                examples.append("Generate an insightful chart from this data")
                
            return examples
            
        except Exception as e:
            logger.error(f"Error in generate_example_prompts: {str(e)}")
            return [
                "Show me trends over time",
                "Create a bar chart comparing categories",
                "Visualize the distribution of values"
            ]