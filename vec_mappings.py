import numpy as np
import os
from scipy.misc import imread
import constants
from io import BytesIO

def load_single_from_binary(data):
    img = imread(BytesIO(data))
    im_shape = img.shape
    if len(im_shape)>2:
        #convert to gray img
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        
        #convert ot gray, faster
        img = np.mean(img,-1)
    im_shape = img.shape
    #if im_shape[0]== 57 and im_shape[1]==300: 
        # each immage has size 57x300, we have to 
        # resize images to 64x304, because conv nets work better with 
        # dimensions divided by 2, we add 4 pixesl at the top
        # 3 pixesl at the bottom and 2 to left and right
        # TODO: change it to more automatic way, what if image size will be 
        # different than 57.300?
        #im_pad = np.pad(img,((4,3),(2,2)), 'constant', constant_values=(255,))
        #im_shape = im_pad.shape
    if im_shape[0]== constants.IMG_RAWHEIGHT and im_shape[1]==constants.IMG_RAWWIDTH:
        im_pad = np.pad(img,((0,0),(4,4)), 'constant', constant_values=(255,))
        im_shape = im_pad.shape
        X = im_pad.flatten()
        return 1,X
    else:
        print('\n-->Image={} with different dimensions then {}*{}, shape={}, skip it'.format(file,constants.IMG_HEIGHT,constants.IMG_WIDTH,im_shape))
        return 0,0

def load_single(path):
    with open(path, 'rb') as f:
        read_data = f.read()
    success, X = load_single_from_binary(read_data)
    if (not success):
        return 0,0,0

    # file with padded text with '_', each captach has 6 char lenght
    file = path.split('/')[-1]
    captcha_text = file.split('.')[0].ljust(6,'_')
    #print(captcha_text)

    Y = map_words2vec(captcha_text)
    return (1,X,Y)

def load_dataset(folder="./img", max_files=float('inf'), file_list=[]):
    # number of files
    N = len(file_list)
    if (0 == N):
        raise Exception('file_list is empty')
    
    N = min(max_files,N)
    
    # stores images #N x 19456 (64*304)
    print('load_dataset: loading {} images '.format(N))
    X = np.zeros([N, constants.IMG_HEIGHT*constants.IMG_WIDTH])
    
    # max captcha text = CAPTCHA_LENGTH chars, at each postion coulb be 0...9A..Za..z_ so 63 different chars
    Y = np.zeros([N, constants.CAPTCHA_LENGTH*63])
    captchas = list()
    
    
    for i, file in enumerate(file_list):
        
        if(i>=N):
            break
        
        if (0 == i%100):
            print('{}/{} {}'.format(i,N,file))
        path=folder+'/'+file
        
        success, X0, Y0 = load_single(path)
        
        if (not success):
            continue
        captchas.append(file)
        X[i,:] = X0
        Y[i,:] = Y0
        
    return (X,Y,captchas)
   
   
def random_batch(X,Y, batch_size=128):
    
    shape_X = X.shape
    shape_Y = Y.shape
    
    if shape_X[0]!=shape_Y[0]:
        raise ValueError('X and Y has different number of examples')
        
    num_ele = shape_X[0]
    
    if batch_size > num_ele:
        raise ValueError('Batch cant be larger then X has rows, batch_size={}, num_ele={}'.format(batch_size, num_ele))
        
    
    rand_idx = np.random.choice(num_ele,batch_size, replace=False)
    
    return X[rand_idx,:], Y[rand_idx,:],rand_idx
    

def map_char2pos(c):
    """
    Helper function for one-hot vector char encoding, we have 63 chars each index coresponds
    to particular char
    
    returns index/position of char passed in argument for sequence
    '0...9A...Za...z_'
    char 0 has an index 0
    char A has an index 10 etc.
    
    Arguments
    ==========
    c - single char
    
    return
    ==========
    position in 63 dimensional vector, at which index 
    
    """
    
    if c =='_':
        k=62
        return k
    
    # compute index for digits '0'...'9', '0' has 0 index, '9' has 9
    # asci code for '0' is 48, so when we substract 48 from the char code we should
    # compute proper index for digits 0...9
    k= ord(c)-48
    
    if k>9:
        # char is not digit
        
        # we are computing index for 'A'...'Z'
        # code for char 'A' is 65, 'Z' is 90, and computing indexes for big letters
        # starts from 10 to 35
        # so we have to substract 65-10=55
        k=ord(c)-55
        if  k>35:
            # char is not big letter
            
            # we are computing index for 'a'...'z'
            # code for char 'a' is 97, 'z' is 122, and computing indexes for small letters
            # starts from 36 to 61
            # so we have to substract 97-36=61
            k=ord(c)-61
            if k >61:
                raise ValueError('Wrong character {} its code={}'.format(c,k)) 
    
    return k    
    
    
def map_words2vec(word):
    """
    Maps word of max CAPTCHA_LENGTH characters to vector, each character could be
    '0...9A...Za...z_'
    char 0 has an index 0
    char A has an index 10 etc.
    The vector contains only 0 and 1, each char position in word is encoded by 63 
    continous vector positions, index 0 encodes occurence of char '0' at firs postion,
    index 63 encodes occurence of char '0' at second position in the word
    
    eg. word = 'at' looks like
    
    vec[36]=1
    vec[118]=1
    
    because, at first position we have char 'a' so
    vec[0:9]=0 it is reserved for digits
    vec[10:35] is reserved for big latin letters
    vec[36:61] is reserved for small latin letters, firs in alphabet is 'a' so it is at 36 position
    vec[62] - reserved for '_'


    character at second place 't'
    vec[63:72] - digits
    vec[73:98] - big letters
    vec[99:124] - small letters, 't' is 19 letter in alphabet so 99+19=118
    vec[125] - '_'
    
    Params
    =======
    words: string
    
    Return
    =======
    numpy array, 1260 dim
    
    """
    
    word_len = len(word)
    
    vector = np.zeros(constants.CAPTCHA_LENGTH*63)
    
    if word_len>constants.CAPTCHA_LENGTH:
        raise ValueError('Word={} has length={}, words should have length up to {}'.format(word, word_len, constants.CAPTCHA_LENGTH))
    
    for i, c in enumerate(word):
        idx=i*63+map_char2pos(c)
        
        vector[idx]=1
      
    return vector  
    
    
    
    
def map_vec2words(vec):
    
    # if vec.shape[0]!= 1260:
    #     raise ValueError('vector should has 1260 dims')
        
    chars_pos = vec.nonzero()[0]
    
    return map_vec_pos2words(chars_pos)
    
  
    
    
    
def map_vec_pos2words(chars_pos):
    '''
    
    
    Arguments
    ==========
    chars_pos - array-like, contains position of each char, eg. [10, 5, ...], represents at 0 postion char A, 1-st char 5, whole word is A5
    '''

    word=list()
    
    for i, c in enumerate(chars_pos):
        char_at_pos = i #c/63
        char_idx = c%63
        
        if char_idx<10:
            char_code= char_idx+ ord('0')
        elif char_idx <36:
            char_code= char_idx-10 + ord('A')
        elif char_idx < 62:
            char_code= char_idx-36 + ord('a')
        elif char_idx == 62:
            char_code = ord('_')
        else:
            raise ValueError('not recognized char code')
            
        
        word.append(chr(char_code))
      
    return "".join(word)
    
    
def model_file_name(ds_name, scale_weights):
    return './models/model_{}_init_{}.ckpt'.format(ds_name,scale_weights)
