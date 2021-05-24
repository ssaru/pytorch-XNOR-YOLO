import torch


def yolo_loss(output: torch.Tensor, target: torch.Tensor):
    # lambda_coord = 5
    # lambda_noobj = 0.5

    # # check batch size
    # b, _, _, _ = target.shape
    # _, _, _, n = output.shape

    # # output tensor slice
    # # output tensor shape is [batch, 7, 7, 5 + classes]
    # objness1_output = output[:, :, :, 0]
    # x_offset1_output = output[:, :, :, 1]
    # y_offset1_output = output[:, :, :, 2]
    # width_ratio1_output = output[:, :, :, 3]
    # height_ratio1_output = output[:, :, :, 4]
    # class_output = output[:, :, :, 5:]

    # num_cls = class_output.shape[-1]

    # # label tensor slice
    # objness_label = target[:, :, :, 0]
    # x_offset_label = target[:, :, :, 1]
    # y_offset_label = target[:, :, :, 2]
    # width_ratio_label = target[:, :, :, 3]
    # height_ratio_label = target[:, :, :, 4]
    # class_label = one_hot(class_output, target[:, :, :, 5], device)

    # noobjness_label = torch.neg(torch.add(objness_label, -1))

    # obj_coord1_loss = lambda_coord * \
    #                   torch.sum(objness_label *
    #                     (torch.pow(x_offset1_output - x_offset_label, 2) +
    #                                 torch.pow(y_offset1_output - y_offset_label, 2)))

    # obj_size1_loss = lambda_coord * \
    #                  torch.sum(objness_label *
    #                            (torch.pow(width_ratio1_output - torch.sqrt(width_ratio_label), 2) +
    #                             torch.pow(height_ratio1_output - torch.sqrt(height_ratio_label), 2)))

    # objectness_cls_map = objness_label.unsqueeze(-1)

    # for i in range(num_cls - 1):
    #     objectness_cls_map = torch.cat((objectness_cls_map, objness_label.unsqueeze(-1)), 3)

    # obj_class_loss = torch.sum(objectness_cls_map * torch.pow(class_output - class_label, 2))

    # noobjness1_loss = lambda_noobj * torch.sum(noobjness_label * torch.pow(objness1_output - objness_label, 2))
    # objness1_loss = torch.sum(objness_label * torch.pow(objness1_output - objness_label, 2))

    # total_loss = (obj_coord1_loss + obj_size1_loss + noobjness1_loss + objness1_loss + obj_class_loss)
    # total_loss = total_loss / b

    # return total_loss, obj_coord1_loss / b, obj_size1_loss / b, obj_class_loss / b, noobjness1_loss / b, objness1_loss / b

    return True
