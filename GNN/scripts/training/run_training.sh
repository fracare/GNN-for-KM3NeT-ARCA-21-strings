#!/bin/bash

orcanet train --list_file ${scripts_dir}/${list_file} --config_file ${scripts_dir}/${config_file} --model_file ${scripts_dir}/${model_file} --to_epoch ${no_of_epochs} ${output_dir}
