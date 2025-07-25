from keras.models import Model
from keras import layers, metrics
from keras.optimizers import Adam
from config import img_height, img_width, num_classes_exp, eps
import numpy as np 


def se_block(input_tensor, ratio=8, name=None):
    """
    Squeeze-and-Excitation block melhorado para TensorFlow 2.x/Keras.
    
    Args:
        input_tensor: Tensor de entrada (feature map).
        ratio: Fator de redução de canais (default 8 como no paper original).
        name: Prefixo para nomes das camadas (opcional).
    
    Returns:
        Tensor com atenção recalibrada.
    """
    # Obter número de canais/filtros
    filters = input_tensor.shape[-1]
    se_shape = (1, 1, filters)
    
    # Squeeze: Global Average Pooling
    se = layers.GlobalAveragePooling2D(name=f'{name}_gap' if name else None)(input_tensor)
    se = layers.Reshape(se_shape, name=f'{name}_reshape' if name else None)(se)
    
    # Excitation: Two FC layers with ReLU and Sigmoid
    se = layers.Dense(filters // ratio, 
                     activation='relu',
                     kernel_initializer='he_normal',
                     use_bias=False,
                     name=f'{name}_fc1' if name else None)(se)
    se = layers.Dense(filters, 
                     activation='sigmoid',
                     kernel_initializer='he_normal',
                     use_bias=False,
                     name=f'{name}_fc2' if name else None)(se)
    
    # Scale: Multiply input with excitation weights
    x = layers.Multiply(name=f'{name}_scale' if name else None)([input_tensor, se])
    
    return x

def H( inputs, num_filters , dropout_rate,use_se): #adicionando use_se por causa do opt
    x = layers.BatchNormalization( epsilon=eps )( inputs )
    x = layers.Activation('relu')(x)

    out_conv = []
    for i in [(1,1),(3,3),(5,5),(0,0)]:
        p = x
        if i == (1,1):
                p = layers.Conv2D(num_filters, (1,1), padding="same",activation="relu")(p)
                out_conv.append(layers.Conv2D(num_filters, (1,1), padding="same",activation="relu")(p))
        elif i == (0,0):
                p = layers.MaxPool2D(pool_size=(2, 2), padding="same",strides=(1,1))(p)
                out_conv.append(layers.Conv2D(num_filters, (1,1), padding="same",activation="relu")(p))
        else:
                p = layers.Conv2D(num_filters, (1,1), padding="same",activation="relu")(p)
                p = layers.SeparableConv2D(num_filters, i, padding="same",activation="relu")(p)
                out_conv.append(layers.SeparableConv2D(num_filters, i, padding="same",activation="relu")(p))
                
    
    x = layers.concatenate(out_conv, axis = -1)
    # Adicionando o SE condicionalmente
    if use_se:
        x = se_block(x, ratio=8, name=None)

    x = layers.Dropout(rate=dropout_rate )(x)
    return x

def transition(inputs, num_filters , compression_factor , dropout_rate, use_se):
    # compression_factor is the 'θ'
    x = layers.BatchNormalization( epsilon=eps )(inputs)
    x = layers.Activation('relu')(x)
    num_feature_maps = inputs.shape[1] # The value of 'm'

    x = layers.Conv2D(int(np.floor(num_feature_maps * compression_factor)) ,
                        kernel_size=(1, 1), use_bias=False, padding='same' ,
                        kernel_initializer='he_normal')(x)
    x = layers.Dropout(rate=dropout_rate)(x)

    # adicionando atencao SE condicionalmente
    if use_se:
        x = se_block(x, ratio=8, name=None)

    x = layers.AveragePooling2D(pool_size=(2, 2))(x)
    return x

def dense_block( inputs, num_layers, num_filters, growth_rate , dropout_rate,block_idx,use_se_in_H ):
    for i in range(num_layers): # num_layers is the value of 'l'
        conv_outputs = H(inputs, num_filters , dropout_rate,use_se=use_se_in_H ) # por causa do use_s em H
        inputs = layers.Concatenate()([conv_outputs, inputs])
        num_filters += growth_rate # To increase the number of filters for each layer.
    return inputs, num_filters


def get_model(input_shape,
           num_blocks,
           num_layers_per_block,
           growth_rate,
           dropout_rate,
           compress_factor,
           num_filters,
           num_classes,
           se_config): # passamos se_config para definir ele
    
    # Determinar onde colocar os Se_blocks baseado na combinação do opt
    use_se_in_H = se_config in ['apenas_H','transicao_e_H','H_e_topo', 'todas']
    use_se_in_transition = se_config in ['apenas_transicao', 'transicao_e_H', 'transicao_e_topo', 'todas']
    use_se_in_final = se_config in ['apenas_topo','transicao_e_topo','H_e_topo','todas']

    inputs = layers.Input( shape=input_shape )
    x = layers.Conv2D( num_filters , kernel_size=( 3 , 3 ) , padding="same", use_bias=False, kernel_initializer='he_normal')( inputs )
    for i in range( num_blocks ):
        x, num_filters = dense_block(x, num_layers_per_block , num_filters, growth_rate , dropout_rate,block_idx=i,use_se_in_H=use_se_in_H)
        x = transition(x, num_filters , compress_factor , dropout_rate,use_se=use_se_in_transition)
        
    # x = cbam_block(x, ratio=8, name="cbam_final")
    # x = se_block(x,ratio=8, name="se_final")
    if use_se_in_final:
        x = se_block(x, ratio=8,name="se_final")
    x = layers.GlobalAveragePooling2D()( x )
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense( num_classes )( x )
    outputs = layers.Activation( 'softmax' )( x )

    model = Model( inputs , outputs )
    
    #model.compile( loss='categorical_crossentropy' ,optimizer=Adam(learning_rate=0.001),
    #                metrics=[ 'accuracy',
    #                          metrics.Recall(thresholds=0.5, class_id=0,name='r_normal'),
    #                          metrics.Recall(thresholds=0.5, class_id=1,name='r_covid'),
    #                          metrics.Recall(thresholds=0.5, class_id=2,name='r_viral')])
    metrics_list = ['accuracy']
    for i in range(num_classes):
        metrics_list.append(
            metrics.Recall(thresholds=0.5, class_id=i, name=f'r_class_{i}')
        )

    model.compile(loss='categorical_crossentropy',
                 optimizer=Adam(learning_rate=0.001),
                 metrics=metrics_list)

    return model