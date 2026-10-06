import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import ConvexHull, QhullError

def plot_variant(builder, save_path=False, kind='bar'):
    if kind not in ('bar', 'swarm'):
        raise ValueError(f"Unknown kind '{kind}'. Available: ['bar', 'swarm']")
    if builder.variant_df is None:
        raise ValueError("No speakers selected yet — call build() first.")

    if len(builder.metric_cols) == 1:
        fig = _plot_1d(builder, kind)
    elif len(builder.metric_cols) == 2:
        fig = _plot_2d(builder)
    else:
        raise ValueError(
            f"plot_variant() only supports 1 or 2 metric columns, got {len(builder.metric_cols)}: "
            f"{builder.metric_cols}. Selection/clustering still works via build()."
        )

    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()

def _plot_1d(builder, kind='bar'):
    metric_col = builder.metric_cols[0]
    target_speaker = builder.target_speaker
    plot_data = builder.per_speaker_df.sort_values(metric_col)
    if builder.cluster_labels is not None:
        plot_data = plot_data.assign(_cluster=builder.cluster_labels.loc[plot_data.index])
    plot_data = plot_data.reset_index(drop=True)
    selected_speakers = builder.variant_df[builder.speaker_col].tolist()

    palette = {'Target Speaker': 'red', 'Selected Speakers': 'salmon', 'Unselected Speakers': 'lightblue'}
    category = [
        'Target Speaker' if speaker == target_speaker
        else 'Selected Speakers' if speaker in selected_speakers
        else 'Unselected Speakers'
        for speaker in plot_data[builder.speaker_col]
    ]

    line_patch = None
    if kind == 'bar':
        fig = plt.figure(figsize=(35, 15))
        plt.bar(
            plot_data[builder.speaker_col].astype(str), plot_data[metric_col],
            color=[palette[c] for c in category], width=0.8,
        )

        if not target_speaker and builder.num_buckets:
            plot_data['_bucket'] = builder.bin(
                plot_data[metric_col], builder.num_buckets, builder.bucket_method
            )
            current_bucket = plot_data['_bucket'].iloc[0]
            for i, bucket_num in enumerate(plot_data['_bucket']):
                if bucket_num != current_bucket:
                    plt.axvline(x=i - 0.5, color='black', linestyle='--', linewidth=2, alpha=0.7)
                    current_bucket = bucket_num
            line_patch = plt.Line2D(
                [0], [0], color='black', linestyle='--', linewidth=2, label='Bucket Boundaries'
            )
        elif not target_speaker and builder.cluster_labels is not None:
            current_cluster = plot_data['_cluster'].iloc[0]
            for i, cluster_num in enumerate(plot_data['_cluster']):
                if cluster_num != current_cluster:
                    plt.axvline(x=i - 0.5, color='black', linestyle='--', linewidth=2, alpha=0.7)
                    current_cluster = cluster_num
            line_patch = plt.Line2D(
                [0], [0], color='black', linestyle='--', linewidth=2, label='Cluster Boundaries'
            )

        plt.xlabel('Speaker ID', fontsize=24)
        plt.xticks(rotation=90, fontsize=15)
    else:
        fig = plt.figure(figsize=(14, 10))
        sns.swarmplot(x=plot_data[metric_col], hue=category, palette=palette, size=6, dodge=False, legend=False)

        if not target_speaker and builder.num_buckets:
            _, bin_edges = builder.bin(
                plot_data[metric_col], builder.num_buckets, builder.bucket_method, retbins=True
            )
            for edge in bin_edges[1:-1]:
                plt.axvline(x=edge, color='black', linestyle='--', linewidth=2, alpha=0.7)
            line_patch = plt.Line2D(
                [0], [0], color='black', linestyle='--', linewidth=2, label='Bucket Boundaries'
            )
        elif not target_speaker and builder.cluster_labels is not None:
            cluster_bounds = plot_data.groupby('_cluster')[metric_col].agg(['min', 'max', 'mean']).sort_values('mean')
            for i in range(len(cluster_bounds) - 1):
                edge = (cluster_bounds['max'].iloc[i] + cluster_bounds['min'].iloc[i + 1]) / 2
                plt.axvline(x=edge, color='black', linestyle='--', linewidth=2, alpha=0.7)
            line_patch = plt.Line2D(
                [0], [0], color='black', linestyle='--', linewidth=2, label='Cluster Boundaries'
            )

        plt.yticks([])
        plt.ylabel('')

    target_speaker_patch = mpatches.Patch(color='red', label='Target Speaker')
    selected_patch = mpatches.Patch(color='salmon', label='Selected Speakers')
    unselected_patch = mpatches.Patch(color='lightblue', label='Unselected Speakers')

    if target_speaker:
        handles = [target_speaker_patch, selected_patch, unselected_patch]
    else:
        handles = [selected_patch, unselected_patch] + ([line_patch] if line_patch else [])
    plt.legend(
        handles=handles, loc='upper left', fontsize=18, handlelength=1.5, handleheight=1.5,
    )

    label = metric_col.replace('_', ' ').title()
    if kind == 'bar':
        plt.ylabel(label, fontsize=24)
        plt.yticks(fontsize=15)
    else:
        plt.xlabel(label, fontsize=24)
        plt.xticks(fontsize=15)
    plt.title(f'{label} per speaker', fontsize=26)
    plt.tight_layout()

    return fig

def _plot_2d(builder):
    x_col, y_col = builder.metric_cols
    target_speaker = builder.target_speaker
    plot_data = builder.per_speaker_df
    selected_speakers = builder.variant_df[builder.speaker_col].tolist()

    colors = [
        'red' if speaker == target_speaker
        else 'salmon' if speaker in selected_speakers
        else 'lightblue'
        for speaker in plot_data[builder.speaker_col]
    ]

    fig = plt.figure(figsize=(15, 15))
    plt.scatter(plot_data[x_col], plot_data[y_col], c=colors, s=80)

    line_patch = None
    if not target_speaker and builder.num_buckets:
        _, x_edges = builder.bin(plot_data[x_col], builder.num_buckets, builder.bucket_method, retbins=True)
        _, y_edges = builder.bin(plot_data[y_col], builder.num_buckets, builder.bucket_method, retbins=True)
        for edge in x_edges[1:-1]:
            plt.axvline(x=edge, color='black', linestyle='--', linewidth=1.5, alpha=0.6)
        for edge in y_edges[1:-1]:
            plt.axhline(y=edge, color='black', linestyle='--', linewidth=1.5, alpha=0.6)
        line_patch = plt.Line2D(
            [0], [0], color='black', linestyle='--', linewidth=2, label='Bucket Boundaries'
        )
    elif not target_speaker and builder.cluster_labels is not None:
        ax = plt.gca()
        # Small fallback circle for clusters too small/degenerate for a hull (<3 points,
        # or collinear points), so every cluster still gets a visible boundary.
        small_radius = 0.02 * max(
            plot_data[x_col].max() - plot_data[x_col].min(),
            plot_data[y_col].max() - plot_data[y_col].min(),
        )
        for cluster_id in range(len(builder.cluster_centers_)):
            member_points = plot_data.loc[builder.cluster_labels == cluster_id, [x_col, y_col]].values
            if len(member_points) == 0:
                continue
            hull_vertices = None
            if len(member_points) >= 3:
                try:
                    hull_vertices = member_points[ConvexHull(member_points).vertices]
                except QhullError:
                    pass  # degenerate (e.g. collinear) cluster — fall back to a circle
            if hull_vertices is not None:
                ax.add_patch(mpatches.Polygon(
                    hull_vertices, closed=True, fill=False,
                    edgecolor='black', linestyle='--', linewidth=1.5, alpha=0.6,
                ))
            else:
                # cluster_centers_ live in cluster_cols' (possibly standardized) space, which
                # can differ from x_col/y_col's actual-value space — use the member points'
                # own centroid instead so the circle lines up with what's plotted.
                ax.add_patch(mpatches.Circle(
                    member_points.mean(axis=0), small_radius, fill=False,
                    edgecolor='black', linestyle='--', linewidth=1.5, alpha=0.6,
                ))
        line_patch = plt.Line2D(
            [0], [0], color='black', linestyle='--', linewidth=2, label='Cluster Boundaries'
        )

    target_speaker_patch = mpatches.Patch(color='red', label='Target Speaker')
    selected_patch = mpatches.Patch(color='salmon', label='Selected Speakers')
    unselected_patch = mpatches.Patch(color='lightblue', label='Unselected Speakers')
    if target_speaker:
        handles = [target_speaker_patch, selected_patch, unselected_patch]
    else:
        handles = [selected_patch, unselected_patch] + ([line_patch] if line_patch else [])
    plt.legend(handles=handles, loc='upper left', fontsize=14)

    plt.xlabel(x_col.replace('_', ' ').title(), fontsize=18)
    plt.ylabel(y_col.replace('_', ' ').title(), fontsize=18)
    plt.title(f'{x_col} vs {y_col} per speaker', fontsize=20)
    plt.tight_layout()

    return fig

