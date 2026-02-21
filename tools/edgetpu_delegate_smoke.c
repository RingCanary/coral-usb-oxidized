#include <stdio.h>
#include <stdlib.h>

// Minimal declarations from edgetpu_c.h so this smoke tool does not depend on
// TensorFlow Lite headers at compile time.
typedef struct TfLiteDelegate TfLiteDelegate;

enum edgetpu_device_type {
  EDGETPU_APEX_PCI = 0,
  EDGETPU_APEX_USB = 1,
};

struct edgetpu_device {
  enum edgetpu_device_type type;
  const char* path;
};

struct edgetpu_option {
  const char* name;
  const char* value;
};

extern struct edgetpu_device* edgetpu_list_devices(size_t* num_devices);
extern void edgetpu_free_devices(struct edgetpu_device* dev);
extern TfLiteDelegate* edgetpu_create_delegate(enum edgetpu_device_type type,
                                               const char* name,
                                               const struct edgetpu_option* options,
                                               size_t num_options);
extern void edgetpu_free_delegate(TfLiteDelegate* delegate);
extern const char* edgetpu_version(void);
extern void edgetpu_verbosity(int verbosity);

static const char* device_type_name(enum edgetpu_device_type type) {
  switch (type) {
    case EDGETPU_APEX_PCI:
      return "pci";
    case EDGETPU_APEX_USB:
      return "usb";
    default:
      return "unknown";
  }
}

int main(void) {
  size_t num_devices = 0;
  struct edgetpu_device* devices = edgetpu_list_devices(&num_devices);

  printf("edgetpu_version=%s\n", edgetpu_version());
  printf("device_count=%zu\n", num_devices);

  if (devices == NULL || num_devices == 0) {
    fprintf(stderr, "No EdgeTPU devices found\n");
    if (devices != NULL) {
      edgetpu_free_devices(devices);
    }
    return 2;
  }

  for (size_t i = 0; i < num_devices; i++) {
    const char* path = devices[i].path ? devices[i].path : "(null)";
    printf("device[%zu]: type=%s path=%s\n", i, device_type_name(devices[i].type), path);
  }

  // Verbose logs can help correlate runtime events with usbmon traces.
  edgetpu_verbosity(10);

  TfLiteDelegate* delegate = edgetpu_create_delegate(
      devices[0].type, devices[0].path, NULL, 0);
  if (delegate == NULL) {
    fprintf(stderr, "edgetpu_create_delegate failed\n");
    edgetpu_free_devices(devices);
    return 3;
  }

  printf("delegate_create=ok\n");
  edgetpu_free_delegate(delegate);
  printf("delegate_free=ok\n");

  edgetpu_free_devices(devices);
  return 0;
}
