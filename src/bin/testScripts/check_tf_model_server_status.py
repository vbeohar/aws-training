from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from grpc.beta import implementations
import time
import grpc
import sys

RPC_TIMEOUT = 600.0
CHANNEL_WAIT_TIMEOUT = 5.0
WAIT_FOR_SERVER_READY_INT_SECS = 60
NUM_PREDICTIONS = 5

def WaitForServerReady(port):
  """Waits for a server on the localhost to become ready."""
  for _ in range(0, WAIT_FOR_SERVER_READY_INT_SECS):
    time.sleep(1)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'intentionally_missing_model'

    try:
      # Send empty request to missing model
      channel = implementations.insecure_channel('localhost', port)
      stub = prediction_service_pb2.beta_create_PredictionService_stub(
          channel)
      stub.Predict(request, RPC_TIMEOUT)
    except grpc.RpcError as error:
      # Missing model error will have details containing 'Servable'
      if 'Servable' in error.details():
        print ('Server is ready')
        break

if __name__ == '__main__':
    port = int(sys.argv[1])
    WaitForServerReady(port)