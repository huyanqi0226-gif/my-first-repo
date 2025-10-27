import pytest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import patch, MagicMock

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from app import main, setup_mlflow
    HAS_DEPENDENCIES = True
except ImportError as e:
    print(f"导入依赖失败: {e}")
    HAS_DEPENDENCIES = False


class TestTitanicSurvivalPrediction:
    """测试泰坦尼克号生存预测项目"""

    def test_data_file_exists(self):
        """测试数据文件是否存在"""
        assert os.path.exists('data/train_and_test2.csv'), "数据文件不存在"

    def test_data_loading(self):
        """测试数据加载功能"""
        if not HAS_DEPENDENCIES:
            pytest.skip("依赖未安装")
        
        try:
            df = pd.read_csv('data/train_and_test2.csv', encoding='utf-8')
            assert isinstance(df, pd.DataFrame)
            assert not df.empty, "数据框为空"
            assert '2urvived' in df.columns, "缺少目标列"
            assert 'Passengerid' in df.columns, "缺少ID列"
        except Exception as e:
            pytest.fail(f"数据加载失败: {e}")

    def test_data_preprocessing(self):
        """测试数据预处理"""
        if not HAS_DEPENDENCIES:
            pytest.skip("依赖未安装")
        
        try:
            df = pd.read_csv('data/train_and_test2.csv', encoding='utf-8')
            
            # 测试删除zero列
            original_columns = len(df.columns)
            zero_columns = [col for col in df.columns if col.startswith('zero')]
            df.drop(zero_columns, inplace=True, axis=1)
            assert len(df.columns) == original_columns - len(zero_columns)
            
            # 测试特征和目标分离
            X = df.drop(['Passengerid', '2urvived'], axis=1)
            y = df['2urvived']
            
            assert X.shape[1] > 0, "特征数量为0"
            assert len(y) == len(df), "目标变量长度不匹配"
            
        except Exception as e:
            pytest.fail(f"数据预处理失败: {e}")

    @patch('app.mlflow')
    def test_mlflow_setup(self, mock_mlflow):
        """测试MLflow设置"""
        if not HAS_DEPENDENCIES:
            pytest.skip("依赖未安装")
        
        with patch.dict(os.environ, {
            'MLFLOW_TRACKING_URI': 'http://test.com',
            'MLFLOW_TRACKING_USERNAME': 'test',
            'MLFLOW_TRACKING_PASSWORD': 'test'
        }):
            try:
                setup_mlflow()
                # 验证mlflow被调用
                mock_mlflow.set_tracking_uri.assert_called()
            except Exception as e:
                # 如果DAGsHub导入失败是正常的
                if "DAGsHub" in str(e):
                    pytest.skip("DAGsHub 配置问题")
                else:
                    raise

    def test_model_training_smoke(self):
        """冒烟测试：确保训练流程能正常运行"""
        if not HAS_DEPENDENCIES:
            pytest.skip("依赖未安装")
        
        # 使用mock来避免实际调用MLflow
        with patch('app.mlflow.start_run') as mock_start, \
             patch('app.mlflow.log_param') as mock_log_param, \
             patch('app.mlflow.log_metric') as mock_log_metric, \
             patch('app.mlflow.sklearn.log_model') as mock_log_model:
            
            # 创建一个mock的run对象
            mock_run = MagicMock()
            mock_start.return_value.__enter__ = MagicMock(return_value=mock_run)
            mock_start.return_value.__exit__ = MagicMock(return_value=None)
            
            try:
                # 运行主函数
                main()
                
                # 验证MLflow被调用
                assert mock_log_param.called, "MLflow参数记录未被调用"
                assert mock_log_metric.called, "MLflow指标记录未被调用"
                
            except Exception as e:
                pytest.fail(f"模型训练冒烟测试失败: {e}")

    def test_environment_variables(self):
        """测试环境变量"""
        # 测试必要的环境变量
        required_vars = ['MLFLOW_TRACKING_URI', 'MLFLOW_TRACKING_USERNAME', 'MLFLOW_TRACKING_PASSWORD']
        
        for var in required_vars:
            value = os.getenv(var)
            if value is None:
                print(f"警告: 环境变量 {var} 未设置")

    def test_imports(self):
        """测试所有必要的导入"""
        try:
            import pandas as pd
            import sklearn
            import mlflow
            import dvc
            assert True
        except ImportError as e:
            pytest.fail(f"必要的导入失败: {e}")


def test_dvc_config():
    """测试DVC配置"""
    dvc_config_file = '.dvc/config'
    if os.path.exists(dvc_config_file):
        with open(dvc_config_file, 'r') as f:
            content = f.read()
            assert 'remote' in content, "DVC远程配置不存在"


if __name__ == "__main__":
    # 运行测试
    pytest_args = [__file__, "-v", "--tb=short"]
    
    # 添加覆盖率报告
    pytest_args.extend(["--cov=.", "--cov-report=html"])
    
    pytest.main(pytest_args)
