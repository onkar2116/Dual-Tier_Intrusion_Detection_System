import os
import glob
import pandas as pd
import numpy as np
from src.utils.config import resolve_path


# NSL-KDD column names (41 features + label + difficulty)
NSL_KDD_COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

# NSL-KDD attack type to category mapping
NSL_KDD_ATTACK_MAP = {
    'normal': 'Normal',
    # DoS attacks
    'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS',
    'smurf': 'DoS', 'teardrop': 'DoS', 'mailbomb': 'DoS', 'apache2': 'DoS',
    'processtable': 'DoS', 'udpstorm': 'DoS',
    # Probe attacks
    'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe', 'satan': 'Probe',
    'mscan': 'Probe', 'saint': 'Probe',
    # R2L attacks
    'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L', 'multihop': 'R2L',
    'phf': 'R2L', 'spy': 'R2L', 'warezclient': 'R2L', 'warezmaster': 'R2L',
    'sendmail': 'R2L', 'named': 'R2L', 'snmpgetattack': 'R2L', 'snmpguess': 'R2L',
    'xlock': 'R2L', 'xsnoop': 'R2L', 'worm': 'R2L',
    # U2R attacks
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R', 'rootkit': 'U2R',
    'httptunnel': 'U2R', 'ps': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R',
}

# Category to numeric mapping
CATEGORY_MAP = {
    'Normal': 0, 'DoS': 1, 'Probe': 2, 'R2L': 3, 'U2R': 4
}


class DataLoader:
    """Load datasets for the IDS system. Supports NSL-KDD, CICIDS2017, and synthetic data."""

    def load(self, config):
        """Load dataset based on config."""
        dataset_name = config['dataset']['primary']
        if dataset_name == 'NSL-KDD':
            path = resolve_path(config['dataset']['path'])
            return self.load_nsl_kdd(path)
        elif dataset_name == 'CICIDS2017':
            path = resolve_path(config['dataset']['secondary_path'])
            return self.load_cicids2017(path)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    def load_nsl_kdd(self, data_dir):
        """
        Load NSL-KDD dataset.

        Expects KDDTrain+.txt and KDDTest+.txt in data_dir.
        Returns a combined DataFrame with proper column names and attack categories.
        """
        train_path = os.path.join(data_dir, 'KDDTrain+.txt')
        test_path = os.path.join(data_dir, 'KDDTest+.txt')

        if not os.path.exists(train_path):
            raise FileNotFoundError(
                f"NSL-KDD dataset not found at {data_dir}. "
                f"Please download KDDTrain+.txt and KDDTest+.txt from "
                f"https://www.unb.ca/cic/datasets/nsl.html and place them in {data_dir}"
            )

        df_train = pd.read_csv(train_path, header=None, names=NSL_KDD_COLUMNS)
        df_train['split'] = 'train'

        if os.path.exists(test_path):
            df_test = pd.read_csv(test_path, header=None, names=NSL_KDD_COLUMNS)
            df_test['split'] = 'test'
            df = pd.concat([df_train, df_test], ignore_index=True)
        else:
            df = df_train

        # Drop difficulty column
        df = df.drop(columns=['difficulty'], errors='ignore')

        # Map attack names to categories
        df['attack_category'] = df['label'].map(
            lambda x: NSL_KDD_ATTACK_MAP.get(x, 'Unknown')
        )

        return df

    def load_cicids2017(self, data_dir):
        """
        Load CICIDS2017 dataset from CSV files.

        Expects CSV files in data_dir.
        """
        csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in {data_dir}. "
                f"Please download CICIDS2017 from https://www.unb.ca/cic/datasets/ids-2017.html"
            )

        dfs = []
        for f in csv_files:
            df = pd.read_csv(f)
            dfs.append(df)

        df = pd.concat(dfs, ignore_index=True)
        df.columns = df.columns.str.strip()

        # Map label to attack_category
        df['attack_category'] = df['Label'].apply(
            lambda x: 'Normal' if x == 'BENIGN' else x
        )

        return df

    def generate_synthetic(self, n_samples=1000, n_features=41, random_seed=42):
        """
        Generate synthetic data matching NSL-KDD feature structure.
        Useful for testing without real datasets.
        """
        np.random.seed(random_seed)

        # Numeric features
        numeric_data = np.random.randn(n_samples, n_features - 3).astype(np.float32)
        numeric_data = np.abs(numeric_data)  # IDS features are mostly non-negative

        # Categorical features
        protocols = np.random.choice(['tcp', 'udp', 'icmp'], n_samples)
        services = np.random.choice(['http', 'ftp', 'smtp', 'ssh', 'dns', 'telnet', 'other'], n_samples)
        flags = np.random.choice(['SF', 'S0', 'REJ', 'RSTR', 'RSTO', 'SH', 'S1', 'S2', 'S3', 'OTH'], n_samples)

        # Labels: 60% normal, 40% attack (split among categories)
        labels = np.random.choice(
            ['Normal', 'DoS', 'Probe', 'R2L', 'U2R'],
            n_samples,
            p=[0.6, 0.2, 0.1, 0.06, 0.04]
        )

        # Build DataFrame with NSL-KDD column names
        numeric_cols = [c for c in NSL_KDD_COLUMNS if c not in ['protocol_type', 'service', 'flag', 'label', 'difficulty']]
        df = pd.DataFrame(numeric_data, columns=numeric_cols[:n_features - 3])

        df.insert(1, 'protocol_type', protocols)
        df.insert(2, 'service', services)
        df.insert(3, 'flag', flags)
        df['label'] = labels
        df['attack_category'] = labels

        return df
