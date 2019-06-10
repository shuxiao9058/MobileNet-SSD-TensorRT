#include "common.h"
#include "cudaUtility.h"
#include "mathFunctions.h"
#include "pluginImplement.h"
#include "tensorNet.h"
#include "loadImage.h"
#include "imageBuffer.h"
#include <chrono>
#include <thread>


const char* model  = "../../model/MobileNetSSD_deploy_iplugin.prototxt";
const char* weight = "../../model/MobileNetSSD_deploy.caffemodel";

const char* INPUT_BLOB_NAME = "data";

const char* OUTPUT_BLOB_NAME = "detection_out";
static const uint32_t BATCH_SIZE = 1;

//image buffer size = 10
//dropFrame = false
ConsumerProducerQueue<cv::Mat> *imageBuffer = new ConsumerProducerQueue<cv::Mat>(10,false);

class Timer {
public:
    void tic() {
        start_ticking_ = true;
        start_ = std::chrono::high_resolution_clock::now();
    }
    void toc() {
        if (!start_ticking_)return;
        end_ = std::chrono::high_resolution_clock::now();
        start_ticking_ = false;
        t = std::chrono::duration<double, std::milli>(end_ - start_).count();
        //std::cout << "Time: " << t << " ms" << std::endl;
    }
    double t;
private:
    bool start_ticking_ = false;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
    std::chrono::time_point<std::chrono::high_resolution_clock> end_;
};


/* *
 * @TODO: unifiedMemory is used here under -> ( cudaMallocManaged )
 * */
float* allocateMemory(DimsCHW dims, char* info)
{
    float* ptr;
    size_t size;
    std::cout << "Allocate memory: " << info << std::endl;
    size = BATCH_SIZE * dims.c() * dims.h() * dims.w();
    assert(!cudaMallocManaged( &ptr, size*sizeof(float)));
    return ptr;
}


void loadImg( cv::Mat &input, int re_width, int re_height, float *data_unifrom,const float3 mean,const float scale )
{
    int i;
    int j;
    int line_offset;
    int offset_g;
    int offset_r;
    cv::Mat dst;

    unsigned char *line = NULL;
    float *unifrom_data = data_unifrom;

    cv::resize( input, dst, cv::Size( re_width, re_height ), (0.0), (0.0), cv::INTER_LINEAR );
    offset_g = re_width * re_height;
    offset_r = re_width * re_height * 2;
    for( i = 0; i < re_height; ++i )
    {
        line = dst.ptr< unsigned char >( i );
        line_offset = i * re_width;
        for( j = 0; j < re_width; ++j )
        {
            // b
            unifrom_data[ line_offset + j  ] = (( float )(line[ j * 3 ] - mean.x) * scale);
            // g
            unifrom_data[ offset_g + line_offset + j ] = (( float )(line[ j * 3 + 1 ] - mean.y) * scale);
            // r
            unifrom_data[ offset_r + line_offset + j ] = (( float )(line[ j * 3 + 2 ] - mean.z) * scale);
        }
    }
}

//thread read video
void readPicture()
{
    cv::VideoCapture cap("../../testVideo/test.avi");
    cv::Mat image;
    while(cap.isOpened())
    {
        cap >> image;
        imageBuffer->add(image);
    }
}

int main(int argc, char *argv[])
{
    std::vector<std::string> output_vector = {OUTPUT_BLOB_NAME};
    TensorNet tensorNet;
    tensorNet.LoadNetwork(model, weight, INPUT_BLOB_NAME, output_vector, BATCH_SIZE);

    DimsCHW dimsData = tensorNet.getTensorDims(INPUT_BLOB_NAME);
    DimsCHW dimsOut = tensorNet.getTensorDims(OUTPUT_BLOB_NAME);

    float *data = allocateMemory(dimsData, (char *)"input blob");
    std::cout << "allocate data" << std::endl;
    float *output = allocateMemory(dimsOut, (char *)"output blob");
    std::cout << "allocate output" << std::endl;
    int height = 300;
    int width = 300;

    cv::Mat frame, srcImg;

    void *imgCPU;
    void *imgCUDA;
    Timer timer;

    //    std::string imgFile = "../../testPic/test.jpg";
    //    frame = cv::imread(imgFile);
    std::thread readTread(readPicture);
    readTread.detach();

    int failCount = 0;
    int frameCount = 0;

    float sumSecs = 0.0;
    int sumImgs = 0;
    int emptyFrameCount = 0;
    //    int64 start=0,end=0;

    while (1)
    {
        imageBuffer->consume(frame);

        if (frame.empty())
        {
            if (failCount <= 10)
            {
                failCount++;
                continue;
            }
            break;
        }

        failCount = 0;
        frameCount++;
        srcImg = frame.clone();

        // Start timer
        double cvTimer = (double)cv::getTickCount();

        cv::resize(frame, frame, cv::Size(300, 300));
        const size_t size = width * height * sizeof(float3);

        if (CUDA_FAILED(cudaMalloc(&imgCUDA, size)))
        {
            cout << "Cuda Memory allocation error occured." << endl;
            return false;
        }

        void *imgData = malloc(size);
        memset(imgData, 0, size);

        loadImg(frame, height, width, (float *)imgData, make_float3(127.5, 127.5, 127.5), 0.007843);
        cudaMemcpyAsync(imgCUDA, imgData, size, cudaMemcpyHostToDevice);

        void *buffers[] = {imgCUDA, output};

        timer.tic();
        tensorNet.imageInference(buffers, output_vector.size() + 1, BATCH_SIZE);
        timer.toc();
        double msTime = timer.t;

        vector<vector<float>> detections;

        for (int k = 0; k < 100; k++)
        {
            if (output[7 * k + 1] == -1)
                break;
            float classIndex = output[7 * k + 1];
            float confidence = output[7 * k + 2];
            float xmin = output[7 * k + 3];
            float ymin = output[7 * k + 4];
            float xmax = output[7 * k + 5];
            float ymax = output[7 * k + 6];
            //        std::cout << classIndex << " , " << confidence << " , "  << xmin << " , " << ymin<< " , " << xmax<< " , " << ymax << std::endl;
            int x1 = static_cast<int>(xmin * srcImg.cols);
            int y1 = static_cast<int>(ymin * srcImg.rows);
            int x2 = static_cast<int>(xmax * srcImg.cols);
            int y2 = static_cast<int>(ymax * srcImg.rows);
            cv::rectangle(srcImg, cv::Rect2f(cv::Point(x1, y1), cv::Point(x2, y2)), cv::Scalar(255, 0, 255), 1);
        }
        float fps = cv::getTickFrequency() / ((double)cv::getTickCount() - cvTimer);
        float secs = ((double)cv::getTickCount() - cvTimer) / cv::getTickFrequency();
        sumSecs += secs;
        sumImgs += 1;
        // Display FPS on frame
        std::cout << std::fixed << std::setprecision(2)
                  << "Computing time: " << secs * 1000
                  << "ms, FPS: " << fps << std::endl;

        //    cv::imshow("mobileNet",srcImg);
        //   cv::waitKey(40);
        //    std::string fileName = "./out/" + sprintf
        std::ostringstream stringStream;
        stringStream << "./out/" << std::setw(8) << std::setfill('0') << std::fixed << frameCount << ".jpg";
        std::string copyOfStr = stringStream.str();

        //std::string result = fmt::sprintf("./out/%05d.jpg", frameCount);
        imwrite(copyOfStr, srcImg);

        free(imgData);
    }

    if (sumImgs > 0)
    {
        std::cout << "AVG Predicted time: " << std::setprecision(5)
                  << sumSecs / sumImgs * 1000 << "ms, FPS: " << 1 / (sumSecs / sumImgs) << std::endl;
    }
    cudaFree(imgCUDA);
    cudaFreeHost(imgCPU);
    cudaFree(output);
    tensorNet.destroy();
    return 0;
}
