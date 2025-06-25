import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple
import json
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


def extract_document_lengths_and_phenotypes(
    implied_data: Dict,
) -> List[Tuple[str, int, int, int]]:
    """
    Extract document lengths and phenotype counts from implied_phenotypes.json

    Args:
        implied_data: The implied phenotypes JSON data

    Returns:
        List of tuples: (case_id, text_length, direct_count, implied_count)
    """
    document_data = []

    for case_id, case_data in implied_data.items():
        if "text" not in case_data:
            continue

        text = case_data["text"]
        text_length = len(text)
        # print(case_id)
        # print(text)
        # Count direct and implied phenotypes
        direct_count = 0
        implied_count = 0

        # Count direct phenotypes
        if "direct_phenotypes" in case_data:
            direct_count = len(case_data["direct_phenotypes"])

        # Count implied phenotypes
        if "implied_phenotypes_with_context" in case_data:
            implied_count = len(case_data["implied_phenotypes_with_context"])
        print(implied_count)
        # Only include documents that have phenotypes
        total_phenotypes = direct_count + implied_count
        if total_phenotypes > 0:
            document_data.append((case_id, text_length, direct_count, implied_count))

    return document_data


def create_phenotype_area_chart(
    document_data: List[Tuple[str, int, int, int]],
    output_dir: str = None,
    smoothing: bool = True,
) -> go.Figure:
    """
    Create area chart showing phenotype type distribution by document length

    Args:
        document_data: List of (case_id, text_length, direct_count, implied_count)
        output_dir: Directory to save chart (optional)
        smoothing: Whether to apply smoothing to the data

    Returns:
        Plotly figure object
    """
    # Convert to DataFrame
    df = pd.DataFrame(
        document_data,
        columns=["case_id", "text_length", "direct_count", "implied_count"],
    )

    # Calculate percentages
    df["total_phenotypes"] = df["direct_count"] + df["implied_count"]
    df["direct_percentage"] = (df["direct_count"] / df["total_phenotypes"]) * 100
    df["implied_percentage"] = (df["implied_count"] / df["total_phenotypes"]) * 100

    # Sort by text length for proper area chart
    df = df.sort_values("text_length").reset_index(drop=True)

    # Apply smoothing if requested (moving average)
    if smoothing and len(df) > 10:
        window_size = max(3, len(df) // 20)  # Adaptive window size
        df["direct_percentage_smooth"] = (
            df["direct_percentage"]
            .rolling(window=window_size, center=True, min_periods=1)
            .mean()
        )
        df["implied_percentage_smooth"] = (
            df["implied_percentage"]
            .rolling(window=window_size, center=True, min_periods=1)
            .mean()
        )
        direct_y = df["direct_percentage_smooth"]
        implied_y = df["implied_percentage_smooth"]
    else:
        direct_y = df["direct_percentage"]
        implied_y = df["implied_percentage"]

    # Use the same color scheme as your pie charts
    direct_color = "rgb(103,122,165)"  # Grayish-blue for Direct
    implied_color = "rgb(252,141,98)"  # Orange for Implied

    # Create area chart
    fig = go.Figure()

    # Add direct phenotypes area (bottom to direct percentage)
    fig.add_trace(
        go.Scatter(
            x=df["text_length"],
            y=direct_y,
            fill="tozeroy",
            mode="lines",
            name="Direct Phenotypes",
            line=dict(color=direct_color, width=0),
            fillcolor=direct_color,
            hovertemplate="<b>Direct Phenotypes</b><br>"
            + "Document Length: %{x:,} characters<br>"
            + "Percentage: %{y:.1f}%<br>"
            + "<extra></extra>",
        )
    )

    # Add implied phenotypes area (from direct percentage to 100%)
    # Create array of 100s that matches the length of the data
    hundred_percent = [100] * len(df)

    fig.add_trace(
        go.Scatter(
            x=df["text_length"],
            y=hundred_percent,  # ✅ Now this is an array
            fill="tonexty",  # Fill to next y (which is the direct percentage line)
            mode="lines",
            name="Implied Phenotypes",
            line=dict(color=implied_color, width=0),
            fillcolor=implied_color,
            hovertemplate="<b>Implied Phenotypes</b><br>"
            + "Document Length: %{x:,} characters<br>"
            + "Percentage: %{customdata:.1f}%<br>"
            + "<extra></extra>",
            customdata=implied_y,
        )
    )

    # Update layout
    fig.update_layout(
        title=dict(
            text="Phenotype Type Distribution by Document Length",
            x=0.5,
            font=dict(size=20),
        ),
        xaxis=dict(
            title="Document Length (characters)",
            title_font=dict(size=14),
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            title="Percentage of Phenotypes",
            title_font=dict(size=14),
            tickfont=dict(size=12),
            range=[0, 100],
            ticksuffix="%",
        ),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.02,
            font=dict(size=12),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
        ),
        width=1000,
        height=600,
        margin=dict(t=80, b=60, l=80, r=60),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")

    # Save as PNG if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        png_path = os.path.join(output_dir, "phenotype_distribution_by_length.png")
        fig.write_image(png_path, width=1000, height=600, scale=2)
        print(f"Saved area chart to: {png_path}")

    return fig


def create_phenotype_length_analysis(
    implied_data_path: str, output_dir: str = None
) -> Dict:
    """
    Complete analysis of phenotype distribution by document length

    Args:
        implied_data_path: Path to implied_phenotypes.json file
        output_dir: Directory to save charts and analysis

    Returns:
        Dictionary with analysis results and figure
    """
    # Load data
    print("Loading implied phenotypes data...")
    with open(implied_data_path, "r") as f:
        implied_data = json.load(f)

    # Extract document data
    print("Extracting document lengths and phenotype counts...")
    document_data = extract_document_lengths_and_phenotypes(implied_data)

    if not document_data:
        print("No valid document data found!")
        return {}

    # Create DataFrame for analysis
    df = pd.DataFrame(
        document_data,
        columns=["case_id", "text_length", "direct_count", "implied_count"],
    )
    df["total_phenotypes"] = df["direct_count"] + df["implied_count"]
    df["direct_percentage"] = (df["direct_count"] / df["total_phenotypes"]) * 100
    df["implied_percentage"] = (df["implied_count"] / df["total_phenotypes"]) * 100

    # Print summary statistics
    print(f"\n" + "=" * 60)
    print("📊 PHENOTYPE DISTRIBUTION BY DOCUMENT LENGTH")
    print("=" * 60)
    print(f"Total documents analyzed: {len(df)}")
    print(
        f"Document length range: {df['text_length'].min():,} - {df['text_length'].max():,} characters"
    )
    print(f"Average document length: {df['text_length'].mean():.0f} characters")
    print(f"Median document length: {df['text_length'].median():.0f} characters")
    print(f"\nTotal phenotypes: {df['total_phenotypes'].sum()}")
    print(f"Average phenotypes per document: {df['total_phenotypes'].mean():.1f}")
    print(
        f"Overall direct percentage: {(df['direct_count'].sum() / df['total_phenotypes'].sum() * 100):.1f}%"
    )
    print(
        f"Overall implied percentage: {(df['implied_count'].sum() / df['total_phenotypes'].sum() * 100):.1f}%"
    )

    # Length-based analysis
    print(f"\n📏 LENGTH-BASED PATTERNS:")

    # Quartile analysis
    q1, q2, q3 = df["text_length"].quantile([0.25, 0.5, 0.75])

    short_docs = df[df["text_length"] <= q1]
    medium_short_docs = df[(df["text_length"] > q1) & (df["text_length"] <= q2)]
    medium_long_docs = df[(df["text_length"] > q2) & (df["text_length"] <= q3)]
    long_docs = df[df["text_length"] > q3]

    print(
        f"Short documents (≤{q1:.0f} chars): {len(short_docs)} docs, {short_docs['direct_percentage'].mean():.1f}% direct"
    )
    print(
        f"Medium-short documents ({q1:.0f}-{q2:.0f} chars): {len(medium_short_docs)} docs, {medium_short_docs['direct_percentage'].mean():.1f}% direct"
    )
    print(
        f"Medium-long documents ({q2:.0f}-{q3:.0f} chars): {len(medium_long_docs)} docs, {medium_long_docs['direct_percentage'].mean():.1f}% direct"
    )
    print(
        f"Long documents (>{q3:.0f} chars): {len(long_docs)} docs, {long_docs['direct_percentage'].mean():.1f}% direct"
    )

    # Create area chart
    print(f"\n📈 Creating area chart...")
    fig = create_phenotype_area_chart(document_data, output_dir, smoothing=True)

    # Create summary statistics
    results = {
        "total_documents": len(df),
        "length_stats": {
            "min": int(df["text_length"].min()),
            "max": int(df["text_length"].max()),
            "mean": float(df["text_length"].mean()),
            "median": float(df["text_length"].median()),
        },
        "phenotype_stats": {
            "total_phenotypes": int(df["total_phenotypes"].sum()),
            "avg_per_document": float(df["total_phenotypes"].mean()),
            "overall_direct_pct": float(
                df["direct_count"].sum() / df["total_phenotypes"].sum() * 100
            ),
            "overall_implied_pct": float(
                df["implied_count"].sum() / df["total_phenotypes"].sum() * 100
            ),
        },
        "quartile_analysis": {
            "short": {
                "count": len(short_docs),
                "direct_pct": (
                    float(short_docs["direct_percentage"].mean())
                    if len(short_docs) > 0
                    else 0
                ),
            },
            "medium_short": {
                "count": len(medium_short_docs),
                "direct_pct": (
                    float(medium_short_docs["direct_percentage"].mean())
                    if len(medium_short_docs) > 0
                    else 0
                ),
            },
            "medium_long": {
                "count": len(medium_long_docs),
                "direct_pct": (
                    float(medium_long_docs["direct_percentage"].mean())
                    if len(medium_long_docs) > 0
                    else 0
                ),
            },
            "long": {
                "count": len(long_docs),
                "direct_pct": (
                    float(long_docs["direct_percentage"].mean())
                    if len(long_docs) > 0
                    else 0
                ),
            },
        },
        "figure": fig,
    }

    return results


def create_phenotype_count_area_chart(
    document_data: List[Tuple[str, int, int, int]],
    output_dir: str = None,
    smoothing: bool = True,
) -> go.Figure:
    """
    Create area chart showing phenotype counts (not percentages) by document word count

    Args:
        document_data: List of (case_id, text_length, direct_count, implied_count)
        output_dir: Directory to save chart (optional)
        smoothing: Whether to apply smoothing to the data

    Returns:
        Plotly figure object
    """
    # Convert to DataFrame
    df = pd.DataFrame(
        document_data,
        columns=["case_id", "text_length", "direct_count", "implied_count"],
    )

    # Convert character count to word count (approximate: divide by 5)
    df["word_count"] = df["text_length"] / 5
    df["word_count"] = df["word_count"].round().astype(int)

    # Sort by word count for proper area chart
    df = df.sort_values("word_count").reset_index(drop=True)

    # Apply smoothing if requested (moving average)
    if smoothing and len(df) > 10:
        window_size = max(3, len(df) // 20)  # Adaptive window size
        df["direct_count_smooth"] = (
            df["direct_count"]
            .rolling(window=window_size, center=True, min_periods=1)
            .mean()
        )
        df["implied_count_smooth"] = (
            df["implied_count"]
            .rolling(window=window_size, center=True, min_periods=1)
            .mean()
        )
        direct_y = df["direct_count_smooth"]
        implied_y = df["implied_count_smooth"]
    else:
        direct_y = df["direct_count"]
        implied_y = df["implied_count"]

    # Calculate total for the top of the area chart
    total_y = direct_y + implied_y

    # Use the same color scheme as your pie charts
    direct_color = "rgb(103,122,165)"  # Grayish-blue for Direct
    implied_color = "rgb(252,141,98)"  # Orange for Implied

    # Create area chart
    fig = go.Figure()

    # Add direct phenotypes area (bottom to direct count)
    fig.add_trace(
        go.Scatter(
            x=df["word_count"],
            y=direct_y,
            fill="tozeroy",
            mode="lines",
            name="Direct Phenotypes",
            line=dict(color=direct_color, width=0),
            fillcolor=direct_color,
            hovertemplate="<b>Direct Phenotypes</b><br>"
            + "Document Length: %{x:,} words<br>"
            + "Count: %{y:.0f}<br>"
            + "<extra></extra>",
        )
    )

    # Add implied phenotypes area (from direct count to total count)
    fig.add_trace(
        go.Scatter(
            x=df["word_count"],
            y=total_y,
            fill="tonexty",  # Fill to next y (which is the direct count line)
            mode="lines",
            name="Implied Phenotypes",
            line=dict(color=implied_color, width=0),
            fillcolor=implied_color,
            hovertemplate="<b>Implied Phenotypes</b><br>"
            + "Document Length: %{x:,} words<br>"
            + "Count: %{customdata:.0f}<br>"
            + "<extra></extra>",
            customdata=implied_y,
        )
    )

    # Update layout (no title)
    fig.update_layout(
        xaxis=dict(
            title="Document Length (words)",
            title_font=dict(size=14),
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            title="Number of Phenotypes",
            title_font=dict(size=14),
            tickfont=dict(size=12),
        ),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.02,
            font=dict(size=12),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
        ),
        width=1000,
        height=600,
        margin=dict(t=60, b=60, l=80, r=60),  # Reduced top margin since no title
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")

    # Save as PNG if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        png_path = os.path.join(output_dir, "phenotype_counts_by_word_length.png")
        fig.write_image(png_path, width=1000, height=600, scale=2)
        print(f"Saved area chart to: {png_path}")

    return fig


def create_phenotype_count_area_chart(
    document_data: List[Tuple[str, int, int, int]],
    output_dir: str = None,
    smoothing: bool = True,
) -> go.Figure:
    """
    Create area chart showing phenotype counts (not percentages) by document word count

    Args:
        document_data: List of (case_id, text_length, direct_count, implied_count)
        output_dir: Directory to save chart (optional)
        smoothing: Whether to apply smoothing to the data

    Returns:
        Plotly figure object
    """
    # Convert to DataFrame
    df = pd.DataFrame(
        document_data,
        columns=["case_id", "text_length", "direct_count", "implied_count"],
    )

    # The text_length column should already contain word counts from the extraction function
    # But let's rename it for clarity
    df["word_count"] = df["text_length"]

    # Sort by word count for proper area chart
    df = df.sort_values("word_count").reset_index(drop=True)

    # Apply smoothing if requested (moving average)
    if smoothing and len(df) > 10:
        window_size = max(3, len(df) // 20)  # Adaptive window size
        df["direct_count_smooth"] = (
            df["direct_count"]
            .rolling(window=window_size, center=True, min_periods=1)
            .mean()
        )
        df["implied_count_smooth"] = (
            df["implied_count"]
            .rolling(window=window_size, center=True, min_periods=1)
            .mean()
        )
        direct_y = df["direct_count_smooth"]
        implied_y = df["implied_count_smooth"]
    else:
        direct_y = df["direct_count"]
        implied_y = df["implied_count"]

    # Calculate total for the top of the area chart
    total_y = direct_y + implied_y

    # Use the same color scheme as your pie charts
    direct_color = "rgb(103,122,165)"  # Grayish-blue for Direct
    implied_color = "rgb(252,141,98)"  # Orange for Implied

    # Create area chart
    fig = go.Figure()

    # Add direct phenotypes area (bottom to direct count)
    fig.add_trace(
        go.Scatter(
            x=df["word_count"],
            y=direct_y,
            fill="tozeroy",
            mode="lines",
            name="Direct Phenotypes",
            line=dict(color=direct_color, width=0),
            fillcolor=direct_color,
            hovertemplate="<b>Direct Phenotypes</b><br>"
            + "Word Count: %{x:,}<br>"
            + "Count: %{y:.0f}<br>"
            + "<extra></extra>",
        )
    )

    # Add implied phenotypes area (from direct count to total count)
    fig.add_trace(
        go.Scatter(
            x=df["word_count"],
            y=total_y,
            fill="tonexty",  # Fill to next y (which is the direct count line)
            mode="lines",
            name="Implied Phenotypes",
            line=dict(color=implied_color, width=0),
            fillcolor=implied_color,
            hovertemplate="<b>Implied Phenotypes</b><br>"
            + "Word Count: %{x:,}<br>"
            + "Count: %{customdata:.0f}<br>"
            + "<extra></extra>",
            customdata=implied_y,
        )
    )

    # Update layout (no title)
    fig.update_layout(
        xaxis=dict(
            title="Word Count",
            title_font=dict(size=14),
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            title="Number of Phenotypes",
            title_font=dict(size=14),
            tickfont=dict(size=12),
        ),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.02,
            font=dict(size=12),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
        ),
        width=1000,
        height=600,
        margin=dict(t=60, b=60, l=80, r=60),  # Reduced top margin since no title
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")

    # Save as PNG if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        png_path = os.path.join(output_dir, "phenotype_counts_by_word_length.png")
        fig.write_image(png_path, width=1000, height=600, scale=2)
        print(f"Saved area chart to: {png_path}")

    return fig


def create_phenotype_count_area_chart(
    document_data: List[Tuple[str, int, int, int]],
    output_dir: str = None,
    smoothing: bool = True,
) -> go.Figure:
    """
    Create area chart showing phenotype counts (not percentages) by document word count

    Args:
        document_data: List of (case_id, text_length, direct_count, implied_count)
        output_dir: Directory to save chart (optional)
        smoothing: Whether to apply smoothing to the data

    Returns:
        Plotly figure object
    """
    # Convert to DataFrame
    df = pd.DataFrame(
        document_data,
        columns=["case_id", "text_length", "direct_count", "implied_count"],
    )

    # The text_length column should already contain word counts from the extraction function
    # But let's rename it for clarity
    df["word_count"] = df["text_length"]

    # Sort by word count for proper area chart
    df = df.sort_values("word_count").reset_index(drop=True)

    # Apply smoothing if requested (moving average)
    if smoothing and len(df) > 10:
        window_size = max(3, len(df) // 20)  # Adaptive window size
        df["direct_count_smooth"] = (
            df["direct_count"]
            .rolling(window=window_size, center=True, min_periods=1)
            .mean()
        )
        df["implied_count_smooth"] = (
            df["implied_count"]
            .rolling(window=window_size, center=True, min_periods=1)
            .mean()
        )
        direct_y = df["direct_count_smooth"]
        implied_y = df["implied_count_smooth"]
    else:
        direct_y = df["direct_count"]
        implied_y = df["implied_count"]

    # Calculate total for the top of the area chart
    total_y = direct_y + implied_y

    # Use the same color scheme as your pie charts
    direct_color = "rgb(103,122,165)"  # Grayish-blue for Direct
    implied_color = "rgb(252,141,98)"  # Orange for Implied

    # Create area chart
    fig = go.Figure()

    # Add direct phenotypes area (bottom to direct count)
    fig.add_trace(
        go.Scatter(
            x=df["word_count"],
            y=direct_y,
            fill="tozeroy",
            mode="lines",
            name="Direct Phenotypes",
            line=dict(color=direct_color, width=0),
            fillcolor=direct_color,
            hovertemplate="<b>Direct Phenotypes</b><br>"
            + "Word Count: %{x:,}<br>"
            + "Count: %{y:.0f}<br>"
            + "<extra></extra>",
        )
    )

    # Add implied phenotypes area (from direct count to total count)
    fig.add_trace(
        go.Scatter(
            x=df["word_count"],
            y=total_y,
            fill="tonexty",  # Fill to next y (which is the direct count line)
            mode="lines",
            name="Implied Phenotypes",
            line=dict(color=implied_color, width=0),
            fillcolor=implied_color,
            hovertemplate="<b>Implied Phenotypes</b><br>"
            + "Word Count: %{x:,}<br>"
            + "Count: %{customdata:.0f}<br>"
            + "<extra></extra>",
            customdata=implied_y,
        )
    )

    # Update layout (no title)
    fig.update_layout(
        xaxis=dict(
            title="Word Count",
            title_font=dict(size=28),
            tickfont=dict(size=24),
        ),
        yaxis=dict(
            title="Number of Phenotypes",
            title_font=dict(size=28),
            tickfont=dict(size=24),
        ),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.02,
            font=dict(size=24),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
        ),
        width=1000,
        height=600,
        margin=dict(t=60, b=60, l=80, r=60),  # Reduced top margin since no title
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")

    # Save as PNG if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        png_path = os.path.join(output_dir, "phenotype_counts_by_word_length.png")
        fig.write_image(png_path, width=1000, height=600, scale=2)
        print(f"Saved area chart to: {png_path}")

    return fig


def extract_document_word_lengths_and_phenotypes(
    implied_data: Dict,
) -> List[Tuple[str, int, int, int]]:
    """
    Extract document word lengths and phenotype counts from implied_phenotypes.json

    Args:
        implied_data: The implied phenotypes JSON data

    Returns:
        List of tuples: (case_id, word_count, direct_count, implied_count)
    """
    document_data = []

    for case_id, case_data in implied_data.items():
        if "text" not in case_data:
            continue

        text = case_data["text"]
        # Convert to word count (approximate: split by whitespace)
        word_count = len(text.split())

        # Count direct and implied phenotypes
        direct_count = 0
        implied_count = 0

        # Count direct phenotypes
        if "direct_phenotypes" in case_data:
            direct_count = len(case_data["direct_phenotypes"])

        # Count implied phenotypes
        if "implied_phenotypes_with_context" in case_data:
            implied_count = len(case_data["implied_phenotypes_with_context"])

        # Only include documents that have phenotypes
        total_phenotypes = direct_count + implied_count
        if total_phenotypes > 0:
            document_data.append((case_id, word_count, direct_count, implied_count))

    return document_data


def create_phenotype_word_count_analysis(
    implied_data_path: str, output_dir: str = None
) -> Dict:
    """
    Complete analysis of phenotype counts by document word count

    Args:
        implied_data_path: Path to implied_phenotypes.json file
        output_dir: Directory to save charts and analysis

    Returns:
        Dictionary with analysis results and figure
    """
    # Load data
    print("Loading implied phenotypes data...")
    with open(implied_data_path, "r") as f:
        implied_data = json.load(f)

    # Extract document data with word counts
    print("Extracting document word lengths and phenotype counts...")
    document_data = extract_document_word_lengths_and_phenotypes(implied_data)

    if not document_data:
        print("No valid document data found!")
        return {}

    # Create DataFrame for analysis
    df = pd.DataFrame(
        document_data,
        columns=["case_id", "word_count", "direct_count", "implied_count"],
    )
    df["total_phenotypes"] = df["direct_count"] + df["implied_count"]

    # Print summary statistics
    print(f"\n" + "=" * 60)
    print("📊 PHENOTYPE COUNTS BY DOCUMENT WORD LENGTH")
    print("=" * 60)
    print(f"Total documents analyzed: {len(df)}")
    print(
        f"Document word count range: {df['word_count'].min():,} - {df['word_count'].max():,} words"
    )
    print(f"Average document length: {df['word_count'].mean():.0f} words")
    print(f"Median document length: {df['word_count'].median():.0f} words")
    print(f"\nTotal phenotypes: {df['total_phenotypes'].sum()}")
    print(f"Average phenotypes per document: {df['total_phenotypes'].mean():.1f}")
    print(f"Total direct phenotypes: {df['direct_count'].sum()}")
    print(f"Total implied phenotypes: {df['implied_count'].sum()}")

    # Word count-based analysis
    print(f"\n📏 WORD COUNT-BASED PATTERNS:")

    # Quartile analysis
    q1, q2, q3 = df["word_count"].quantile([0.25, 0.5, 0.75])

    short_docs = df[df["word_count"] <= q1]
    medium_short_docs = df[(df["word_count"] > q1) & (df["word_count"] <= q2)]
    medium_long_docs = df[(df["word_count"] > q2) & (df["word_count"] <= q3)]
    long_docs = df[df["word_count"] > q3]

    print(
        f"Short documents (≤{q1:.0f} words): {len(short_docs)} docs, avg {short_docs['total_phenotypes'].mean():.1f} phenotypes"
    )
    print(
        f"Medium-short documents ({q1:.0f}-{q2:.0f} words): {len(medium_short_docs)} docs, avg {medium_short_docs['total_phenotypes'].mean():.1f} phenotypes"
    )
    print(
        f"Medium-long documents ({q2:.0f}-{q3:.0f} words): {len(medium_long_docs)} docs, avg {medium_long_docs['total_phenotypes'].mean():.1f} phenotypes"
    )
    print(
        f"Long documents (>{q3:.0f} words): {len(long_docs)} docs, avg {long_docs['total_phenotypes'].mean():.1f} phenotypes"
    )

    # Create area chart
    print(f"\n📈 Creating area chart...")
    fig = create_phenotype_count_area_chart(document_data, output_dir, smoothing=True)

    # Create summary statistics
    results = {
        "total_documents": len(df),
        "word_count_stats": {
            "min": int(df["word_count"].min()),
            "max": int(df["word_count"].max()),
            "mean": float(df["word_count"].mean()),
            "median": float(df["word_count"].median()),
        },
        "phenotype_stats": {
            "total_phenotypes": int(df["total_phenotypes"].sum()),
            "total_direct": int(df["direct_count"].sum()),
            "total_implied": int(df["implied_count"].sum()),
            "avg_per_document": float(df["total_phenotypes"].mean()),
        },
        "quartile_analysis": {
            "short": {
                "count": len(short_docs),
                "avg_phenotypes": (
                    float(short_docs["total_phenotypes"].mean())
                    if len(short_docs) > 0
                    else 0
                ),
            },
            "medium_short": {
                "count": len(medium_short_docs),
                "avg_phenotypes": (
                    float(medium_short_docs["total_phenotypes"].mean())
                    if len(medium_short_docs) > 0
                    else 0
                ),
            },
            "medium_long": {
                "count": len(medium_long_docs),
                "avg_phenotypes": (
                    float(medium_long_docs["total_phenotypes"].mean())
                    if len(medium_long_docs) > 0
                    else 0
                ),
            },
            "long": {
                "count": len(long_docs),
                "avg_phenotypes": (
                    float(long_docs["total_phenotypes"].mean())
                    if len(long_docs) > 0
                    else 0
                ),
            },
        },
        "figure": fig,
    }

    return results


# Example usage
def main():
    """Example usage of the phenotype analysis - now with both versions"""

    # Replace with your actual file path
    implied_data_path = "data/dataset/implied_phenotypes.json"
    output_dir = "figs/"

    # Option 1: Run the original percentage-based analysis with character length
    print("Running original percentage-based analysis...")
    results_percentage = create_phenotype_length_analysis(implied_data_path, output_dir)

    if results_percentage:
        print(
            f"\n✅ Percentage analysis complete! Check {output_dir} for the area chart."
        )

    # Option 2: Run the new count-based analysis with word length
    print("\n" + "=" * 60)
    print("Running new count-based analysis with word counts...")
    results_counts = create_phenotype_word_count_analysis(implied_data_path, output_dir)

    if results_counts:
        print(
            f"\n✅ Count analysis complete! Check {output_dir} for the word count area chart."
        )

        # Save analysis results for the word count version
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

            # Save results (excluding the figure for JSON serialization)
            results_for_json = {
                k: v for k, v in results_counts.items() if k != "figure"
            }
            with open(
                os.path.join(output_dir, "word_count_analysis_results.json"), "w"
            ) as f:
                json.dump(results_for_json, f, indent=2)

            print(
                f"Word count analysis results saved to: {os.path.join(output_dir, 'word_count_analysis_results.json')}"
            )


# Alternative: If you only want the word count version, use this simpler main function
def main_word_count_only():
    """Example usage of just the word count phenotype analysis"""

    # Replace with your actual file path
    implied_data_path = "data/dataset/implied_phenotypes.json"
    output_dir = "figs/"

    # Run the new word count-based analysis
    results = create_phenotype_word_count_analysis(implied_data_path, output_dir)

    if results:
        print(
            f"\n✅ Analysis complete! Check {output_dir} for the word count area chart."
        )

        # Save analysis results
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

            # Save results (excluding the figure for JSON serialization)
            results_for_json = {k: v for k, v in results.items() if k != "figure"}
            with open(
                os.path.join(output_dir, "word_count_analysis_results.json"), "w"
            ) as f:
                json.dump(results_for_json, f, indent=2)

            print(
                f"Analysis results saved to: {os.path.join(output_dir, 'word_count_analysis_results.json')}"
            )


if __name__ == "__main__":
    # Choose which version to run:

    # Option 1: Run both analyses (original + new)
    # main()

    # Option 2: Run only the new word count analysis
    main_word_count_only()
