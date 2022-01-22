import cv2
class MyTool(object):
    @staticmethod 
    def cro_img(img,t_height=299,t_width=299):
        a = img.shape
        height=a[0]
        width=a[1]
        if height/width>t_height/t_width:
            height2=t_height/t_width*width
            height2=int(height2)
            n=int((height-height2)/2.0)
            cropImg=img[n:(height2+n), 0:width]
        else:
            width2=t_width*1.0/t_height*height
            width2=int(width2)
            n=int((width-width2)/2.0)
            cropImg=img[ 0:height, n:(width2+n)]
        cropImg=cv2.resize(cropImg,(t_width,t_height),interpolation=cv2.INTER_AREA )
        return cropImg