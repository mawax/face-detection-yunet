using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Dnn;
using Emgu.CV.Structure;
using System.Drawing;

var useWebcam = false;
var pathToVideo = "examples/dancers.mp4";
var renderConfidence = true;

var windowName = "Face detection (Press any key to close)";
CvInvoke.NamedWindow(windowName);

using var capture = useWebcam
    ? new VideoCapture(camIndex: 0)
    : new VideoCapture(fileName: pathToVideo);

using var model = InitializeFaceDetectionModel(new Size(capture.Width, capture.Height));

while (CvInvoke.WaitKey(1) == -1)
{
    var frame = capture.QueryFrame();
    if (frame is null)
    {
        break;
    }

    var faces = new Mat();
    model.Detect(frame, faces);
    DrawDetectedFaces(frame, faces, renderConfidence);

    CvInvoke.Imshow(windowName, frame);
}

FaceDetectorYN InitializeFaceDetectionModel(Size inputSize) => new FaceDetectorYN(
    model: "face_detection_yunet_2022mar.onnx",
    config: string.Empty,
    inputSize: inputSize,
    scoreThreshold: 0.9f,
    nmsThreshold: 0.3f,
    topK: 5000,
    backendId: Emgu.CV.Dnn.Backend.Default,
    targetId: Target.Cpu);

void DrawDetectedFaces(Mat frame, Mat faces, bool renderConfidence)
{
    if (faces.Rows <= 0)
    {
        return;
    }

    // facesData is multidimensional array.
    // The first dimension is the index of the face, the second dimension is the data for that face.
    // The data for each face is 15 elements long:
    //  - the first 4 elements are the bounding box of the face (x, y, width, height)
    //  - the next 10 elements are the x and y coordinates of 5 facial landmarks:
    //      right eye, left eye, nose tip, right mouth corner, left mouth corner
    //  - the last element is the confidence score
    var facesData = (float[,])faces.GetData(jagged: true);

    for (var i = 0; i < facesData.GetLength(0); i++)
    {
        DrawFaceRectangle(frame, (int)facesData[i, 0], (int)facesData[i, 1], (int)facesData[i, 2], (int)facesData[i, 3]);
        DrawFaceLandMarks(frame, i, facesData);

        if (renderConfidence)
        {
            DrawConfidenceText(frame, (int)facesData[i, 0], (int)facesData[i, 1] - 5, facesData[i, 14]);
        }
    }
}

void DrawFaceRectangle(Mat frame, int x, int y, int width, int height)
{
    var faceRectangle = new Rectangle(x, y, width, height);
    CvInvoke.Rectangle(frame, faceRectangle, new MCvScalar(0, 255, 0), 1);
}

void DrawFaceLandMarks(Mat frame, int faceIndex, float[,] facesData)
{
    var landMarkColors = new MCvScalar[]
    {
        new MCvScalar(255, 0, 0),   // right eye
        new MCvScalar(0, 0, 255),   // left eye
        new MCvScalar(0, 255, 0),   // nose tip
        new MCvScalar(255, 0, 255), // right mouth corner
        new MCvScalar(0, 255, 255)  // left mouth corner
    };

    for (var landMark = 0; landMark < 5; landMark++)
    {
        var x = (int)facesData[faceIndex, 4 + landMark * 2];
        var y = (int)facesData[faceIndex, 4 + landMark * 2 + 1];
        CvInvoke.Circle(frame, new Point(x, y), 2, landMarkColors[landMark], -1);
    }
}

void DrawConfidenceText(Mat frame, int x, int y, float confidence)
{
    CvInvoke.PutText(frame, $"{confidence:N4}", new Point(x, y), FontFace.HersheyComplex, 0.3, new MCvScalar(0, 0, 255), 1);
}