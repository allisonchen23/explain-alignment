import argparse
from airium import Airium
from collections import OrderedDict
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, sys
import shutil
from tqdm import tqdm

sys.path.insert(0, 'src')
from utils.utils import ensure_dir, write_json, read_json, read_lists
from utils.visualizations import bar_graph
from utils.df_utils import convert_string_columns

scene_categories = read_lists('data/places365_categories/scene_categories.txt')
group_id_description_dict = {
    "group_1": "Model = Explainer = Human",
    "group_2": "Model = Explainer != Human",
    "group_3": "Model = Human != Explainer",
    "group_4": "Explainer = Human != Model",
    "group_5": "Model != Explainer; Explainer != Human; Human != Model"
}
def copy_file(src_path, 
              dst_dir,
              overwrite=True):
    filename = os.path.basename(src_path)
    assert os.path.isfile(src_path), "Source file {} does not exist.".format(src_path)
    
    ensure_dir(dst_dir)
    dst_path = os.path.join(dst_dir, filename)
    if overwrite or not os.path.isfile(dst_path):
        shutil.copyfile(src_path, dst_path)
    return dst_path
    
def save_pdist(p_dist,
               save_path,
               title=None,
               fig_size=(6, 3)):
    if len(p_dist.shape) == 1:
        p_dist = np.expand_dims(p_dist, axis=0)
    assert len(p_dist.shape) == 2
    
    bar_graph(
        data=p_dist,
        xlabel='Classes',
        ylabel='Unnormalized Probabilities',
        title=title,
        # fig_size=fig_size,
        save_path=save_path)

def multi_bars(data,
               fig,
               axs,
               titles=None,
               xlabels=None,
               ylabels=None,
               fig_size=(4, 4),
               save_path=None,
               show=False):
    n_rows = len(data)
    if titles is not None:
        assert len(titles) == n_rows
    if xlabels is not None:
        assert len(xlabels) == n_rows
    if ylabels is not None:
        assert len(ylabels) == n_rows
    
    # fig, axs = plt.subplots(nrows=n_rows, ncols=1)
    
    for idx, cur_data in enumerate(data):
        n_classes = len(cur_data)
        x_pos = np.arange(n_classes)
        axs[idx].bar(
            x_pos,
            cur_data)
        
        if titles is not None:
            axs[idx].set_title(titles[idx])
        if xlabels is not None:
            axs[idx].set_xlabel(xlabels[idx])
        if ylabels is not None:
            axs[idx].set_ylabel(ylabels[idx])
    # plt.figure(figsize=fig_size)
    fig.set_figheight(fig_size[1])
    fig.set_figwidth(fig_size[0])
    plt.tight_layout()
    if save_path is not None:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path)
    if show:
        plt.show()
     
    return fig, axs
    

def save_assets(csv_paths, 
                asset_paths_path, 
                asset_src_root='data/broden1_224/images',
                overwrite=True,
                sort_column=None,
                debug=True):
    agent_types = ['human', 'explainer', 'model']
    asset_paths = {}
    for csv_idx, csv_path in enumerate(csv_paths):
#         if '.ipynb_checkpoints' in csv_path:
#             continue
            
        group_id = os.path.basename(csv_path).split('.')[0]
        
        print("[{}] Processing file {}/{}".format(datetime.now().strftime(r"%m%d_%H%M%S"), csv_idx+1, len(csv_paths)))
        if debug and csv_idx > 0:
            break
            
        df = pd.read_csv(csv_path)
        asset_save_dir = os.path.dirname(asset_paths_path)
        group_asset_save_dir = os.path.join(asset_save_dir, group_id)
        # image_files = df['filename']
        group_assets = OrderedDict()
        # Convert str -> numpy array
        df = convert_string_columns(
            df, 
            columns=['{}_outputs'.format(agent) for agent in agent_types])
        
        if sort_column is not None:
            df = df.sort_values(sort_column, ascending=False)
        for row_idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
        # for image_idx, image_file in enumerate(tqdm(image_files)):
            image_file = row['filename']
            if debug and row_idx > 5:
                break
            # Get source path
            image_src_path = os.path.join(asset_src_root, image_file)
            # Create destination directory
            image_id = os.path.basename(image_file).split('.')[0]
            image_save_dir = os.path.join(
                group_asset_save_dir, 
                image_id)
            ensure_dir(image_save_dir)
            
            # Copy image
            image_dst_path = copy_file(
                src_path=image_src_path,
                dst_dir=image_save_dir,
                overwrite=overwrite)
            
            # Save unnormalized probability distributions
            data = []
            titles = []
            xlabels = ['Classes' for i in range(len(agent_types))]
            ylabels = ['Unnormalized Probabilities' for i in range(len(agent_types))]
            for agent in agent_types:
                data.append(row['{}_outputs'.format(agent)])
                titles.append('Unnormalized {} probabilities'.format(agent))
            plot_save_path = os.path.join(image_save_dir, 'unnormalized_probabilities.png')
            n_rows = len(data)
            fig, axs = plt.subplots(nrows=n_rows, ncols=1)
            multi_bars(
                data=data,
                fig=fig,
                axs=axs,
                titles=titles,
                xlabels=xlabels,
                ylabels=ylabels,
                fig_size=(6, 6),
                save_path=plot_save_path,
                show=False)
            # Close figure
            plt.close()
            
            # Store asset paths for this file
            group_assets[image_id] = {
                'image_path': image_dst_path,
                'plot_path': plot_save_path,
                'entropy': row['entropy'],
                't2c': row['unnormalized_top_2_confusion'],
                'human_prediction': row['human_predictions'],
                'model_prediction': row['model_predictions'],
                'explainer_prediction': row['explainer_predictions']
            }
        
        asset_paths[group_id] = group_assets
    # asset_save_path = os.path.join(asset_save_dir, 'asset_paths.json')
    write_json(asset_paths, asset_paths_path)
    print("Saved asset paths to {}".format(asset_paths_path))
    return asset_paths
            
def check_assets_saved(csv_paths, asset_paths):
    for csv_path in csv_paths:
        group_id = os.path.basename(csv_path).split('.')[0]
        df = pd.read_csv(csv_path)
        group_assets = asset_paths[group_id]
        if len(df) != len(group_assets):
            return False
    return True
        
def build_html(group_id,
               group_assets,
               html_dir,
               title,
               debug=True):
    
    air = Airium()
    air('<!DOCTYPE html>')
    with air.html(lang="pl"):
        # Set HTML header
        with air.head():
            air.meta(charset="utf-8")
            air.title(_t=title)
            
        # HTML Body
        with air.body():
            # for group_idx, group_id in enumerate(sorted(asset_paths.keys())):
            #     if debug and group_idx > 0:
            #         break
            #     group_assets = asset_paths[group_id]
            
            
            with air.h2():
                air("{}: {}".format(group_id, group_id_description_dict[group_id]))
            # print the idx to category mappings
            with air.h3():
                air("Indices -> Category mappings")
            for idx, scene_category in enumerate(scene_categories):
                air.p(_t="{}: {}".format(idx, scene_category))
            
            # for each image, put image and plots
            for image_idx, (image_id, image_paths_dict) in enumerate(group_assets.items()):
                if debug and image_idx > 5:
                    break
                # Get relative paths for assets
                for key, path in image_paths_dict.items():
                    if 'path' in key:
                        image_paths_dict[key] = os.path.relpath(path, html_dir)
                # Print header and data
                with air.h3():
                    air("{}. {}".format(image_idx+1, image_id))
                air.p(_t='Human entropy: {:.4f} T2C: {:.4f}'.format(
                    image_paths_dict['entropy'], image_paths_dict['t2c']))
                # Print predictions 
                human_pred = image_paths_dict['human_prediction']
                explainer_pred = image_paths_dict['explainer_prediction']
                model_pred = image_paths_dict['model_prediction']
                air.p(_t="Human prediction: {} ({})".format(
                    scene_categories[human_pred], human_pred))
                air.p(_t="Explainer prediction: {} ({})".format(
                    scene_categories[explainer_pred], explainer_pred))
                air.p(_t="Model prediction: {} ({})".format(
                    scene_categories[model_pred], model_pred))
                # Display images and plots
                air.image(src=image_paths_dict['image_path'])
                air.p(_t="\n")
                air.image(src=image_paths_dict['plot_path'])
                air.p(_t="\n\n")
                    
    html_string = str(air)
    return html_string
                    
def save_html(csv_dir, 
               html_dir,
               filenames=None,
               overwrite=False,
               sort_column=None,
               title='index.html',
               debug=True):
    
    # Create list of paths
    if filenames is None or len(filenames) == 0:
        filenames = os.listdir(csv_dir)
    csv_paths = [os.path.join(csv_dir, filename) for filename in filenames if '.ipynb_checkpoints' not in filename]
    csv_paths = sorted(csv_paths)

    # Ensure directory exists
    ensure_dir(html_dir)
    # if len(os.listdir(html_dir)) > 0:
    #     raise ValueError("Directory '{}' is not empty. Please specify a new directory".format(html_dir))
        
    # If assets don't already exist, save assets
    asset_save_dir = os.path.join(html_dir, 'assets')
    asset_paths_path = os.path.join(asset_save_dir, 'asset_paths.json')
    if os.path.exists(asset_save_dir):
        asset_paths = read_json(asset_paths_path)
        assets_saved = check_assets_saved(
            csv_paths=csv_paths,
            asset_paths=asset_paths)
    if not assets_saved or overwrite: #not os.path.exists(asset_save_dir):
        print("Saving assets...")
        asset_paths = save_assets(
            csv_paths=csv_paths,
            asset_paths_path=asset_paths_path,
            overwrite=overwrite,
            sort_column=sort_column,
            debug=debug)
    else:
        print("Loaded assets from {}".format(asset_paths_path))
    
    # Iterate through each group's assets and create an HTML page
    for group_idx, group_id in enumerate(sorted(asset_paths.keys())):
        if debug and group_idx > 0:
            break
        group_assets = asset_paths[group_id]
        html_string = build_html(
            group_id=group_id,
            group_assets=group_assets,
            html_dir=html_dir,
            title=title,
            debug=debug)
    
        html_file_save_path = os.path.join(html_dir, '{}.html'.format(group_id))
        with open(html_file_save_path, 'wb') as f:
            f.write(bytes(html_string, encoding='utf-8'))
        print("Saved HTML file to {}".format(html_file_save_path))
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--csv_dir', type=str, required=True)
    parser.add_argument('--filenames', nargs='+', default=[])
    parser.add_argument('--html_dir', type=str, default='html/ade20k/groups')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    
    save_html(
        csv_dir=args.csv_dir,
        html_dir=args.html_dir,
        filenames=args.filenames,
        overwrite=args.overwrite,
        sort_column=['entropy', 'unnormalized_top_2_confusion'],
        debug=args.debug)