python run_lm.py --epochs=5 --lr=1e-2 --freezing=True --pre_train=converted_fwd.pdparams --save_model_name=freeze_lm.pdparams
python run_lm.py --epochs=150 --lr=1e-3 --pre_train=freeze_lm.pdparams --save_model_name=unfreeze_lm2.pdparams
python run_cls.py --epochs=30 --lr=2e-2 --freezing=-1 --pre_train=unfreeze_lm2.pdparams --save_model_name=cls_freeze_1.pdparams
python run_cls.py --epochs=30 --lr=1e-2 --freezing=-2 --pre_train=cls_freeze_1.pdparams --save_model_name=cls_freeze_2.pdparams
python run_cls.py --epochs=30 --lr=5e-3 --freezing=-3 --pre_train=cls_freeze_2.pdparams --save_model_name=cls_freeze_3.pdparams
python run_cls.py --epochs=30 --lr=1e-3 --pre_train=cls_freeze_3.pdparams --save_model_name=cls_unfreeze.pdparams