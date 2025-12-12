


# cd ./suscape_scenes_10hz
# sed -i 's/2hz/10hz/g' scene-000002/desc.json
# python3 ~/code2/dataset_tools/crop_scene.py regen ./scene-000002 '' '000,100,200,300,400,500,600,700,800,900'
# cp -H -r ../suscape_scenes/scene-000002/label ./scene-000002/label_2hz 
# python ~/code2/SUSTechPoints-be-dev/tools/interpolate_10hz_labels.py --data . --scenes 'scene-000000' 

# 1st argument is the 2hz dataset path
# 2nd argument is the 10hz dataset path
# 3rd argument is scene name


sed -i 's/2hz/10hz/g' $2/$3/desc.json
python3 ~/code2/dataset_tools/crop_scene.py regen  $2/$3 '' '000,100,200,300,400,500,600,700,800,900'
cp -H -r $1/$3/label $2/$3/label_2hz
python ~/code2/SUSTechPoints-be-dev/tools/interpolate_10hz_labels.py --data $2 --scenes $3


# sh ~/code2/SUSTechPoints-be-dev/tools/create_10hz_dataset.sh ../suscape_scenes  . scene-000003 