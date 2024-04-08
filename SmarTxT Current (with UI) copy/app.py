# General Imports and global functions
from flask import Flask, redirect, url_for, request, session, render_template
from flask_bootstrap import Bootstrap
from flask_session import Session
from flask_login import LoginManager, login_user, login_required, UserMixin
import pandas as pd
import numpy as np
import os
from joblib import dump, load
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
import json

app = Flask(__name__)
bootstrap = Bootstrap(app)
app.config['SECRET_KEY'] = 'dogs-are-the-best'
app.config['SESSION_TYPE'] = 'filesystem'
FLASK_DEBUG = True

Session(app)

# Login code
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = ''


@login_manager.user_loader
def load_user(user_id):
    return User(user_id)


class User(UserMixin):
    def __init__(self, id):
        self.id = id


@login_manager.unauthorized_handler
def unauthorized():
    """
    Redirects URL to login page
    """
    return redirect(url_for('login'))


def get_cleaned_df_columns(df):
    """
    Renames df columns to not being capitalized
    """
    new_df = df.copy()
    important_cols = [key for key, _ in session['column_info'].items()]
    column_renaming_dict = {}
    for key, value in session['column_info'].items():
        column_renaming_dict[key] = value['display_name']
    cols_to_keep = [*important_cols, 'Duplicates', 'Topic_Likelihood', 'Best_Topic']
    new_df = new_df[new_df.columns.intersection(cols_to_keep)]
    new_df.rename(columns=column_renaming_dict, inplace=True)
    return new_df


def get_current_df():
    """
    Back up approach to getting current data file.
    This will initially load the raw data file.
    Also, Get the temp_df.pickle when session does not work
    """
    if os.path.exists("temp_df.pickle"):
        df = pd.read_pickle('temp_df.pickle')
    else:
        df = pd.read_pickle(session['dataset_path'])
    return df


def get_topic_top_words(model, feature_names, n_components, n_top_words=2):
    """
    Given a model, list of count vectorizer feature names, number of topics, and number of keywords:
    Return a dataframe with number of keywords for each topic found using LDA

    :param model: LDA Model LDA Model created by top_modeling or online_top_modeling function
    :param feature_names: CountVectorizer feature names CV feature names created from countvectorizer model in
    top_modeling or online_top_modeling functions
    :param n_components: int Number of topics in fitted LDA model, This will always be equal to best components
    :param n_top_words: int Default 10

    """
    topic_names = pd.DataFrame(0, index=range(n_components), columns=['Topic', 'Topic Keywords'])
    for i, topic in enumerate(model.components_):
        topic_names.iloc[i, :] = [i + 1, " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])]
    return topic_names


def apply_topic_modeling(df, num_topics, num_keywords):
    """
    :num_topics: number of topics to be discovered by model
    :num_keywords: number of keywords to represent each topic
    """
    # Initialise the count vectorizer with the English stop words
    count_vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 1), min_df=20)
    # Fit and transform the processed titles
    try:
        count_data = count_vectorizer.fit_transform(df[session['text_col_cleaned']])
        error = None
    except:
        error = "Please broaden your search criteria to include more records"
        return None, None, None, None, error

    # Use randomizedsearchcv to optimize the hyperparameters.
    lda = LDA(n_components=int(num_topics),
              n_jobs=-1,
              random_state=2357)
    lda.fit(count_data)
    topic_feature_names = count_vectorizer.get_feature_names()
    topic_names_df = get_topic_top_words(model=lda,
                                         n_components=int(num_topics),
                                         feature_names=topic_feature_names,
                                         n_top_words=int(num_keywords))
    best_topics = lda.transform(count_data).argmax(axis=1)
    best_prob = lda.transform(count_data).max(axis=1)
    return lda, topic_names_df, best_topics, best_prob, error


def apply_filtering(r, df):
    """
    Function:  allows df to be filtered by the following:
    - Make
    - Model
    - Year
    """
    year = r.form["year"] if r.form["year"] else None
    make = str(r.form["make"]).upper() if r.form["make"] else None
    model = str(r.form["model"]).upper() if r.form["model"] else None
    if year is not None:
        df = df[df['YEARTXT'].astype(str) == year]
        session['year_filter'] = year
        print('filtering on year')
    if make is not None:
        df = df[df['MAKETXT'].astype(str) == make]
        session['make_filter'] = make
        print('filtering on make')
    if model is not None:
        df = df[df['MODELTXT'].astype(str) == model]
        session['model_filter'] = model
        print('filtering on model')
    return df


def reset_selections():
    """
    Function: to standardize and reset filter selections on training page
    """
    session['num_topics'] = 10
    session['num_keywords'] = 10
    session['make_filter'] = None
    session['model_filter'] = None
    session['year_filter'] = None
    session['trained_topic_model'] = False
    session['trained_doc_sim_model'] = False
    if os.path.exists('temp_df.pickle'):
        os.remove('temp_df.pickle')


def reset_dataset_metadata_selections():
    """
    Function: to reset dataset metadata as needed.
    :return:
    """
    session['dataset_name'] = 'Dataset Not Selected'
    session['dataset_description'] = 'Dataset Not Selected'
    session['original_file_name'] = 'Dataset Not Selected'
    session['dataset_id'] = None
    if os.path.exists('temp_df.pickle'):
        os.remove('temp_df.pickle')


def update_session_data(data_dict):
    """
    Given a dataset id, update the session data to mirror that dataset
    :param data_dict: Automatically generated dict from .json based on user selection
    :return: session data saved for later use
    """

    session['dataset_name'] = data_dict['dataset_name']
    session['original_file_name'] = data_dict['original_file_name']
    session['dataset_description'] = data_dict['dataset_description']
    session["id_col"] = data_dict["id_col"]
    session["dataset_path"] = data_dict["dataset_path"]
    session["text_col"] = data_dict["text_col"]
    session["optimal_topics"] = data_dict["optimal_topics"]
    session["optimal_keywords"] = data_dict["optimal_keywords"]
    session["text_col_cleaned"] = data_dict["text_col_cleaned"]
    session["full_dataset_size"] = data_dict["full_dataset_size"]
    session["dupe_removed_size"] = data_dict["dupe_removed_size"]
    session["lookup_table_path"] = data_dict["lookup_table_path"]
    session['image_path'] = data_dict['image_path']
    session["column_info"] = data_dict["column_info"]


def train_models(r, df):
    """
    Function: takes the input from the apply_filtering function and inputs it into the training functions,
    apply_document_similarity and apply_topic_modeling.
    - creates session from previous apply_filtering inputs
    - checks to see top modeling checkbox (tm_checkbox) and document similarity checkbox (ds_checkbox) are checked
    - checks to verify if all data is selected by df length
    """
    num_topics = int(r.form["num_topics"]) if r.form["num_topics"] else session['num_topics']
    num_keywords = int(r.form["num_keywords"]) if r.form["num_keywords"] else session['num_keywords']
    session['num_topics'] = num_topics
    session['num_keywords'] = num_keywords
    tm_checkbox = 'topic_modeling' in request.form
    ds_checkbox = 'doc_sim' in request.form
    error = None
    if tm_checkbox:
        # Our original dataset has 127424 rows. If user has not made any filters
        # Then we should load in saved models instead of fitting new model.
        if df.shape[0] == session["dupe_removed_size"]:
            print('All records detected')
            session['model'] = f'pretrained_topic_model/{session["dataset_id"]}/pretrained_lda.joblib'
            model = load(session['model'])
            num_topics, num_keywords = session["optimal_topics"], session["optimal_keywords"]
            session['num_topics'], session['num_keywords'] = num_topics, num_keywords

            best_topics = load(f'pretrained_topic_model/{session["dataset_id"]}/pretrained_best_topics.joblib')
            df['Best_Topic'] = best_topics
            df['Best_Topic'] += 1
            best_prob = load(f'pretrained_topic_model/{session["dataset_id"]}/pretrained_best_prob.joblib')
            df['Topic_Likelihood'] = best_prob
            keywords = load(f'pretrained_topic_model/{session["dataset_id"]}/pretrained_keywords.joblib')
            session['keywords'] = f'pretrained_topic_model/{session["dataset_id"]}/pretrained_keywords.joblib'
        else:
            model, keywords, topics, best_prob, error = apply_topic_modeling(df, num_topics=num_topics,
                                                                             num_keywords=num_keywords)
            if error is not None:
                return df, error
            df['Best_Topic'] = topics
            # Add 1 to topic to show user 1 to n instead of 0 to n-1
            df['Best_Topic'] += 1
            df['Topic_Likelihood'] = best_prob
            dump(model, 'trained_lda.joblib')
            dump(keywords, 'keyword.joblib')
            session['model'] = 'trained_lda.joblib'
            session['keywords'] = 'keyword.joblib'

        keywords['View Similar Documents'] = ['Topic ' + str(x + 1) for x in range(num_topics)]
        session['keyword_html'] = keywords.to_html(formatters={
            'View Similar Documents': lambda x: f'<a href="{x}">{x}</a>'},
            classes=['table'],
            justify='unset',
            border='inherit',
            escape=False, index=False)

        # checks to see if topic modeling is ran
        session['trained_topic_model'] = True

    if ds_checkbox:

        # checks to see if doc similarity is ran
        session['trained_doc_sim_model'] = True

    else:
        # Need a dummy call to catch if a user doesn't pick a model
        print('Please choose a model to run')

    return df, error


# Route for sending user to login page from initial link
@app.route('/', methods=['GET', 'POST'])
def initial_page():
    """
    Function: Send user to login page if click link localhost:1234/
    """
    return redirect(url_for('login'))


# Route for handling the login page logic
@app.route('/login', methods=['GET', 'POST'])
def login():
    """
    Function: Login page the end user must first come to before entering the app
    """
    error = None
    if request.method == 'POST':
        # Can change this later to admit more people
        if request.form['username'] != 'a' or request.form['password'] != 'a':
            error = 'Invalid Credentials. Please try again.'
        else:
            # Have to identify method, not url
            login_user(User(1))
            reset_selections()
            return redirect(url_for('homepage'))
    return render_template('login.html', error=error)


# Dataset Page
#  TODO make entire home page one route. It works now so not going to break it
@app.route("/home")
@login_required
def homepage():
    reset_dataset_metadata_selections()

    return render_template('homepage.html', dataset_name=session['dataset_name'],
                           dataset_description=session['dataset_description'],
                           original_file_name=session['original_file_name'])


@app.route("/changedataset", methods=['POST'])
@login_required
def changedataset():
    # reset_selections()

    # # User selects a dataset (putting in tsbs as id for testing)
    session["dataset_id"] = request.form['changedataset']
    data_info_file = open(f'{session["dataset_id"]}.json')
    dataset_data = json.load(data_info_file)
    data_info_file.close()
    update_session_data(dataset_data)

    return render_template('homepage.html', dataset_name=session['dataset_name'],
                           dataset_description=session['dataset_description'],
                           original_file_name=session['original_file_name'],
                           filename=session['image_path'])


@app.route("/getfulldfstats", methods=['POST'])
@login_required
def getfulldfstats():
    """
    Function: get stats on home page of full dataset.
    -Methods are called when hitting the 'Get stats' button
    """

    if session['dataset_id'] is None:
        return render_template('homepage.html', dataset_name=session['dataset_name'],
                               dataset_description=session['dataset_description'],
                               original_file_name=session['original_file_name'],
                               error='Select a dataset before getting stats')

    df = get_current_df()
    total_rows = session['full_dataset_size']
    unique_rows = session['dupe_removed_size']
    min_bulletin = df[session['text_col']].str.len().min()
    max_bulletin = df[session['text_col']].str.len().max()
    mean_bulletin = np.round(df[session['text_col']].str.len().mean(), 2)
    return render_template('homepage.html', number_of_rows=total_rows, number_of_unique_rows=unique_rows,
                           min_char=min_bulletin, max_char=max_bulletin, mean_char=mean_bulletin,
                           dataset_name=session['dataset_name'],
                           dataset_description=session['dataset_description'],
                           original_file_name=session['original_file_name'],
                           filename=session['image_path'])


@app.route("/viewdataall", methods=['POST'])
@login_required
def viewdfhead():
    """
   This is just for the dataset page. Will always show the header of the full dataset
   TODO make this a GET since no params are involved
    """

    if session['dataset_id'] is None:
        return render_template('homepage.html', dataset_name=session['dataset_name'],
                               dataset_description=session['dataset_description'],
                               original_file_name=session['original_file_name'],
                               error='Select a dataset before viewing the dataset')
    df = get_current_df()
    df = get_cleaned_df_columns(df)
    # df = df.drop([session["text_col_cleaned"]], axis=1)
    return render_template('view_data.html', dataset_name=session['dataset_name'],
                           data_to_view=df.head(100).to_html())


@app.route("/training_page", methods=['GET', 'POST'])
@login_required
def training_page():
    """
    Note: Without a reset filters option, need to clean session filters each time
    If a user picked "ford focus" and then "chevrolet silverado" no records
    would be returned, because filtering would look for records where both are
    true. By resetting at call, this avoids this issue.
    Verified that this gives intended functionality in app

    Check if a user has pushed any buttons. If not, return the base page
    All of our buttons are attached to the form TrainingPage, so if this hasn't
    been hit then we know a user needs to go to the base.
    Function:
    - verifies selections made and calls reset_selections() to reset them
    - calls apply_filtering to get selections for training
    - get stats for filtered selections
    - allows end user to view filtered data
    - calls train models function
    """
    if session['dataset_id'] is None:
        return render_template('training.html', make_filter=session['make_filter'],
                               model_filter=session['model_filter'],
                               year_filter=session['year_filter'],
                               error='You must select a dataset before training a model')
    try:
        request.form['TrainingPage']
    except KeyError:
        return render_template('training.html', make_filter=session['make_filter'],
                               model_filter=session['model_filter'],
                               year_filter=session['year_filter'])
    reset_selections()
    new_df = get_current_df()
    checkbox = 'all_data' in request.form
    # Only apply filtering if all records is not selected
    if not checkbox:
        new_df = apply_filtering(r=request, df=new_df)
        if new_df.shape[0] == 0:
            return render_template('training.html', make_filter=session['make_filter'],
                                   model_filter=session['model_filter'],
                                   year_filter=session['year_filter'],
                                   error='Invalid Make/Model/Year combination selected.')
        elif new_df.shape[0] < 20:
            return render_template('training.html', make_filter=session['make_filter'],
                                   model_filter=session['model_filter'],
                                   year_filter=session['year_filter'],
                                   error='Too few unique records selected. Please broaden your search criteria.')

    new_df.to_pickle('temp_df.pickle')
    # calling in parameters from training page to verify selections were made - see Note
    getstats_option = 'getstats' in request.form['TrainingPage']
    viewdata_option = 'viewdata' in request.form['TrainingPage']
    trainmodel_option = 'trainmodel' in request.form['TrainingPage']
    # Have 3 options for users within this form, so check the value to see what to call
    if getstats_option:

        # Weird edge case if we don't have duplicates in the other datasets. Leaving in for now
        total_rows = sum(new_df['Duplicates'])
        unique_rows = new_df.shape[0]
        min_bulletin = new_df[session['text_col']].str.len().min()
        max_bulletin = new_df[session['text_col']].str.len().max()
        mean_bulletin = np.round(new_df[session['text_col']].str.len().mean(), 2)
        return render_template('training.html', number_of_rows=total_rows, number_of_unique_rows=unique_rows,
                               min_char=min_bulletin, max_char=max_bulletin, mean_char=mean_bulletin,
                               make_filter=session['make_filter'], model_filter=session['model_filter'],
                               year_filter=session['year_filter'])

    elif viewdata_option:

        view_df = new_df.drop(session['text_col_cleaned'], axis=1).reset_index(drop=True)
        view_df = get_cleaned_df_columns(view_df)
        return render_template('view_data.html', data_to_view=view_df.head(100).to_html())

    elif trainmodel_option:
        new_df, error = train_models(r=request, df=new_df)
        if error is not None:
            return render_template('training.html', training_status='Training Incomplete', error=error,
                                   make_filter=session['make_filter'],
                                   model_filter=session['model_filter'], year_filter=session['year_filter'])
        new_df.to_pickle('temp_df.pickle')
        return render_template('training.html', training_status='Training Finished', make_filter=session['make_filter'],
                               model_filter=session['model_filter'], year_filter=session['year_filter'])
    else:
        # Edge case, but don't think this can ever be hit unintentionally - what is this?
        return render_template('training.html')


# Test topic modeling dynamic pages
@app.route('/Topic <int:topic>', methods=['GET', 'POST'])
@login_required
def best_topic_docs(topic):
    """
    Function: gets best topic from topic modeling output and saves into current df and then as html
    """
    df = get_current_df()

    # Adding one to topic labels in display, so need to subtract one here.
    best_topic_df = df[df['Best_Topic'] == topic]
    best_topic_df = best_topic_df.drop([[session['text_col_cleaned']]], axis=1, errors='ignore')

    best_topic_df = best_topic_df.sort_values(by='Topic_Likelihood', ascending=False).reset_index(drop=True)
    num_rows = min(25, best_topic_df.shape[0])
    best_topic_df = best_topic_df.iloc[:num_rows, :]
    best_topic_df = best_topic_df.drop(['Best_Topic'], axis=1)
    best_topic_df = get_cleaned_df_columns(best_topic_df)
    topic_html = best_topic_df.to_html(classes=['table'], justify='unset')

    keywords = load(session['keywords'])
    topic_keywords = keywords.iloc[topic - 1, 1]
    return render_template('best_topics.html', selected_topic=topic, keyword_list=topic_keywords, topic_html=topic_html)


# Topic Modeling Results Page
@app.route("/topicmodeling")
@login_required
def tm_landing():
    # renders topic modeling output html file from training page
    if session['trained_topic_model']:
        return render_template('topicmodeling.html', test_1=session['keyword_html'])
    else:
        return render_template('topicmodeling.html', error='You must train a topic model before viewing results')


# Document Similarity Results Page
@app.route("/documentsimilarity")
@login_required
def ds_landing():
    if session['trained_doc_sim_model'] is False:
        # returns error if document similarity page is selected w/o training doc similarity model
        return render_template('documentsimilarityerror.html',
                               error='You must train a document similarity model before viewing results')
    try:
        # calls sessions previous filter page
        return render_template('documentsimilarity.html',
                               filtered_make=session['make_filter'],
                               filtered_model=session['model_filter'],
                               filtered_year=session['year_filter'],
                               max_topics=session['num_topics'])
    except KeyError:
        # returns error if topic modeling selection made w/o running topic modeling
        return render_template('documentsimilarity.html',
                               filtered_make=session['make_filter'],
                               filtered_model=session['model_filter'],
                               filtered_year=session['year_filter'],
                               max_topics='No Topic Model Detected')


@app.route("/document_similarity_page", methods=['GET', 'POST'])
@login_required
def document_similarity():
    """
    Function: defines all input parameters for document similarity page.
    -Reference doc (ref_doc): entering a key bulletin number for a document similarity report to be generated against.
    Given a reference doc, the msot similar documents are returned from the generate_report function
    -key phrase (keywords): enter in a key phrase, or string text. Document embedding is created from each model then
    appended
    to overall document embedding and added into a cosine distance matrix. The rank matrix is created again from all
    models'
    cosine distance matrix as done in the save_sim_matrix function.
    -Topic number (topics): filter documents by a specific topic number
    """
    # referance document - entering in a bulletin number
    if request.form['DocSim'] == 'ref_doc':
        ref_doc_no = request.form['ref_document_doc_sim'] if request.form['ref_document_doc_sim'] else None
        if ref_doc_no is not None:

            # Load in most recent lookup table
            lookup_table = pd.read_csv(session['lookup_table_path'], index_col=0)

            # Get current temp_df
            df = get_current_df()

            # Clean up output for similar documents
            df = df.drop([session['text_col_cleaned'], 'Topic_Likelihood'], axis=1, errors='ignore')

            # Filter lookup table by the index shown in current df. This saves time
            lookup_table = lookup_table.loc[df.index, :]

            # Get the index where the user input is the BULNO
            user_index_reference = df.index[df[session['id_col']] == ref_doc_no].tolist()

            if len(user_index_reference) == 0:
                error = 'Invalid Reference Document'
                return render_template('documentsimilarity.html', filtered_make=session['make_filter'],
                                       filtered_model=session['model_filter'], error=error,
                                       filtered_year=session['year_filter'],
                                       max_topics=session['num_topics'])

            # Get the list of indices of similar documents from the lookup table
            all_similar_documents = []
            doc_list = []
            for uir in user_index_reference:
                similar_document_index_to_show = lookup_table.loc[uir, :].values
                similar_document_index_to_show = list(filter(lambda a: a != -1, similar_document_index_to_show))
                all_similar_documents.append(similar_document_index_to_show[:])
                doc_list.append(uir)
            # Need to find the maximum length of a sublist
            max_len = -1
            for asd in all_similar_documents:
                if len(asd) > max_len:
                    max_len = len(asd)

            # Continually add elements to the list by position
            for i in range(max_len):
                for j in all_similar_documents:
                    # Using try and except because not all lists will be the same length
                    # This gives flexibility for variable size lists
                    try:
                        doc_list.append(j[i])
                    except IndexError:
                        continue

            # Only want to show the user n rows and don't repeat the reference document
            doc_list = list(dict.fromkeys(doc_list[:150]))

            # Since we have already applied filters to the temp df previously, we can just check if the similar
            # documents are in the df.index. If they are not, then it is a mismatch in terms of make, model, or year
            # for elem in doc_list:
            #     print(doc_list)
            #     if elem not in df.index:
            #         doc_list.remove(elem)

            doc_list = list(set(doc_list).intersection(set(df.index)))

            doc_list.insert(0, user_index_reference[0])

            # Since we are adding in the user reference at the beginning, we need to remove the other ref to user ref
            duplicate_indices = np.where(np.array(doc_list) == user_index_reference[0])
            index_to_remove = duplicate_indices[0][-1]
            del doc_list[index_to_remove]

            # Filter the original df by the list of indices
            similar_document = df.loc[doc_list, :]

            # Get cleaned column names for similar_document
            similar_document = get_cleaned_df_columns(similar_document)

            # Show the reference document
            ref_doc = similar_document.head(1).to_html()

            # Show every document except the first one in the similarity report
            sim_report = similar_document.iloc[1:, :].to_html()

            print(f'generating report on {session["id_col"]}')
            error = None
        else:
            error = 'Invalid ID entered'
            sim_report = None
            ref_doc = None

        return render_template('documentsimilarity.html', ref_doc=ref_doc,
                               sim_report=sim_report, filtered_make=session['make_filter'],
                               filtered_model=session['model_filter'], error=error,
                               filtered_year=session['year_filter'],
                               max_topics=session['num_topics'])

    # keywords - entering in a key phrase
    if request.form['DocSim'] == 'keywords':

        # Get all keywords and search word by word against the Summaries in the TSBS data
        keyword_list = request.form['keyword_list_doc_sim'].split()
        df = get_current_df()
        search_list = []
        for i in df.index:
            if all(keyword.lower() in str(df[session['text_col']][i]).lower().split() for keyword in keyword_list):
                search_list.append(i)

        # Retrieve indices with search item, sort by duplicates and show.
        sim_report = df[df.index.isin(search_list)]

        if len(sim_report) == 0:
            error = 'Phrase does not exist in selected dataset'
            return render_template('documentsimilarity.html', error=error,
                                   filtered_make=session['make_filter'], filtered_model=session['model_filter'],
                                   filtered_year=session['year_filter'], max_topics=session['num_topics'])

        sim_report = get_cleaned_df_columns(sim_report)
        sim_report = sim_report.drop(['Best_Topic', 'Topic_Likelihood'], axis=1, errors='ignore')
        sim_report = sim_report.sort_values(by='Duplicates', ascending=False)
        ref_doc = sim_report.head(1).to_html()

        return render_template('documentsimilarity.html', ref_doc=ref_doc, sim_report=sim_report.iloc[1:, :].to_html(),
                               filtered_make=session['make_filter'], filtered_model=session['model_filter'],
                               filtered_year=session['year_filter'], max_topics=session['num_topics'])


# logout page
@app.route("/logout", methods=['GET', 'POST'])
@login_required
def logout():
    session.clear()
    if os.path.exists("temp_df.pickle"):
        os.remove('temp_df.pickle')
    if os.path.exists("keyword.joblib"):
        os.remove("keyword.joblib")
    if os.path.exists("trained_lda.joblib"):
        os.remove("trained_lda.joblib")
    return redirect(url_for('login'))


if __name__ == "__main__":
    try:
        session.clear()
    except:
        pass
    if os.path.exists("temp_df.pickle"):
        os.remove('temp_df.pickle')
    if os.path.exists("keyword.joblib"):
        os.remove("keyword.joblib")
    if os.path.exists("trained_lda.joblib"):
        os.remove("trained_lda.joblib")
    app.run(port=1234, debug=True)
